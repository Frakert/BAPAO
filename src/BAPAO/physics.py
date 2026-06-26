"""Physics-based continuum models for dose-dependent film growth.

This module implements finite-difference solvers for transient dose-based
continuum models. Supports unified physics handling all combinations of:
thermal ALD (s0), CVD sticking (s_CVD), recombination (r), and desorption (d0).

Model uses sparse matrix linear algebra for efficient computation.
The unified DoseBasedContinuumUnified model handles all parameter combinations
and is the single supported solver for profile simulation.

Key Classes:
  DoseBasedContinuumUnified: Universal model for all parameter combinations
  PhysicsModel: Unified interface for model selection and evaluation
"""

from __future__ import annotations

import multiprocessing
import os
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def transient_worker(job):
    s0 = job.get("s0", 0.0)
    s0_cvd = job.get("s0_CVD", 0.0)
    r = job.get("r", 0.0)
    d0 = job.get("d0", 0.0)
    k = job.get("k", 100.0)
    ar = job.get("AR", 1000)

    sim = DoseBasedContinuumUnified(
        sticking_prob=s0,
        sticking_prob_cvd=s0_cvd,
        recombination_prob=r,
        desorption_prob=d0,
        dim_dose=k,
        AR=ar,
    )

    result = sim.solve_transient(target_dose=k)
    return job, result


class DoseBasedContinuumUnified:
    """Unified dose-based continuum model for all ALD/CVD physics combinations.

    Solves transient continuum equations with simultaneous support for:
    - Thermal ALD sticking (s0)
    - CVD sticking (s_CVD)
    - Recombination (r)
    - Desorption (d0)

    This model handles all parameter combinations elegantly, with degenerate
    cases naturally occurring when any parameter is zero. Separates ALD
    coverage (bounded [0,1]) from CVD thickness (unbounded), providing
    physically consistent behavior across all regimes.

    Attributes
    ----------
    sticking_prob : float
        Thermal ALD sticking probability (0-1).
    sticking_prob_cvd : float
        CVD sticking probability (0-1).
    recombination_prob : float
        Recombination probability (0-1).
    desorption_prob : float
        Desorption probability (0-1).
    dim_dose : float
        Dimensionless dose parameter.
    AR : int
        Aspect ratio of the feature (geometry parameter).
    xi : np.ndarray, optional
        Spatial grid points after solve_transient, shape (N,).
    theta_ald : np.ndarray, optional
        ALD coverage (bounded) after solve_transient, shape (N,).
    h_cvd : np.ndarray, optional
        CVD thickness (unbounded) after solve_transient, shape (N,).
    theta_total : np.ndarray, optional
        Total thickness (ALD + CVD) after solve_transient, shape (N,).
    """

    def __init__(
        self,
        sticking_prob: float = 0.0,
        sticking_prob_cvd: float = 0.0,
        recombination_prob: float = 0.0,
        desorption_prob: float = 0.0,
        dim_dose: float = 100.0,
        AR: int = 1000,
    ):
        """Initialize DoseBasedContinuumUnified.

        Parameters
        ----------
        sticking_prob : float
            Thermal ALD sticking probability.
        sticking_prob_cvd : float
            CVD sticking probability.
        recombination_prob : float
            Recombination probability.
        desorption_prob : float
            Desorption probability.
        dim_dose : float
            Dimensionless dose parameter.
        AR : int
            Aspect ratio of the feature.
        """
        self.sticking_prob = sticking_prob
        self.sticking_prob_cvd = sticking_prob_cvd
        self.recombination_prob = recombination_prob
        self.desorption_prob = desorption_prob
        self.dim_dose = dim_dose
        self.AR = AR

        self.xi: Optional[NDArray[np.float64]] = None
        self.theta_ald: Optional[NDArray[np.float64]] = None
        self.h_cvd: Optional[NDArray[np.float64]] = None
        self.theta_total: Optional[NDArray[np.float64]] = None

    def _build_transport_matrix_sparse(self, N, dx, free_sites, alpha, beta, nu, AR):
        """Construct sparse transport matrix for unified physics.

        Parameters
        ----------
        N : int
            Number of spatial grid points.
        dx : float
            Spatial grid spacing.
        free_sites : np.ndarray
            Free site fraction (1 - theta_ald), shape (N,).
        alpha : float
            Dimensionless ALD rate coefficient.
        beta : float
            Dimensionless CVD rate coefficient.
        nu : float
            Dimensionless recombination rate coefficient.
        AR : int
            Aspect ratio.

        Returns
        -------
        A : scipy.sparse matrix
            Sparse CSR format matrix.
        b : np.ndarray
            Right-hand side vector with boundary conditions.
        """
        inv_dx2 = 1.0 / dx**2
        loss = alpha * free_sites + beta + nu

        lower = np.full(N - 1, inv_dx2)
        main = -2.0 * inv_dx2 - loss
        upper = np.full(N - 1, inv_dx2)

        main[0] = 1.0
        upper[0] = 0.0

        bc_coeff = (alpha * free_sites[-1] + beta) / (2.0 * AR)
        lower[-1] = -1.0 / dx
        main[-1] = 1.0 / dx + bc_coeff

        A = diags([lower, main, upper], [-1, 0, 1], format="csr")
        b = np.zeros(N)
        b[0] = 1.0
        return A, b

    def solve_transient(
        self,
        target_dose: float,
        N: int = 1000,
        num_time_steps: int = 1000,
    ) -> dict[str, NDArray[np.float64]]:
        """Solve unified transient model to target dose.

        Integrates unified continuum equations with separated ALD coverage
        (bounded) and CVD thickness (unbounded) using implicit finite
        difference scheme.

        Parameters
        ----------
        target_dose : float
            Final dose value to integrate to.
        N : int, optional
            Number of spatial grid points. Default 1000.
        num_time_steps : int, optional
            Number of dose integration steps. Default 1000.

        Returns
        -------
        xi : np.ndarray
            Spatial grid points, shape (N,).
        theta_total : np.ndarray
            Total thickness (ALD + CVD), shape (N,).
        """
        s0 = self.sticking_prob
        s0_cvd = self.sticking_prob_cvd
        r = self.recombination_prob
        d0 = self.desorption_prob
        ar = self.AR

        alpha = (3 / 4) * ar**2 * s0
        beta = (3 / 4) * ar**2 * s0_cvd
        nu = (3 / 4) * ar**2 * r
        delta = (3 / 4) * ar**2 * d0

        xi = np.linspace(0, 1, N)
        dx = xi[1] - xi[0]
        dose_step = target_dose / num_time_steps

        # State variables: separate ALD and CVD
        theta_ald = np.zeros(N)
        h_cvd = np.zeros(N)
        n = np.ones(N)

        current_dose = 0.0

        while current_dose < target_dose:
            # Smooth free-site fraction (only ALD sites are "occupied")
            free_sites = 1.0 - theta_ald

            # Transport solve for precursor concentration
            A_mat, b = self._build_transport_matrix_sparse(N, dx, free_sites, alpha, beta, nu, ar)
            n = spsolve(A_mat, b)

            # ALD (Implicit Euler - unconditionally stable, naturally bounded [0, 1])
            theta_ald = (theta_ald + dose_step * alpha * n) / (
                1.0 + dose_step * alpha * n + dose_step * delta
            )

            # CVD (Implicit Euler - prevents negative oscillation on desorption)
            h_cvd = (h_cvd + dose_step * beta * n) / (1.0 + dose_step * delta)

            # keep both in the physically meaningful range.
            theta_ald = np.clip(theta_ald, 0.0, 1.0)
            h_cvd = np.clip(h_cvd, 0.0, np.inf)

            current_dose += dose_step

        # Store results
        self.xi = xi
        self.theta_ald = theta_ald
        self.h_cvd = h_cvd
        self.theta_total = theta_ald + h_cvd

        result_dict = {
            "xi": self.xi,
            "theta": self.theta_total,
        }

        return result_dict

    def evaluate_at(self, query_points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Interpolate total thickness profile at arbitrary spatial points.

        Parameters
        ----------
        query_points : np.ndarray
            Spatial points for evaluation, shape (M,), values in [0, 1].

        Returns
        -------
        np.ndarray
            Interpolated total thickness values, shape (M,).

        Raises
        ------
        ValueError
            If solve_transient has not been called yet.
        """
        if self.theta_total is None or self.xi is None:
            raise ValueError("No profile solved yet. Run solve_transient first.")
        return np.interp(query_points, self.xi, self.theta_total)

    def run_transient_parallel(
        self, settings_dict_list: list[dict[str, Any]], num_workers: int = 4
    ):
        with multiprocessing.Pool(num_workers) as pool:
            results = list(pool.imap(transient_worker, settings_dict_list, chunksize=1))
        return results


class PhysicsModel:
    """Unified interface for physics-based continuum models.

    Uses the unified continuum solver for all supported parameter
    combinations and evaluates normalized thickness profiles.

    Attributes
    ----------
    AR : int
        Aspect ratio of the feature.
    N : int
        Number of spatial grid points for finite difference discretization.

    Examples
    --------
    >>> model = PhysicsModel(AR=10, N=500)
    >>> xi, theta = model.evaluate(
    ...     xi=np.linspace(0, 1, 100),
    ...     s0=0.8,
    ...     k=100.0
    ... )
    """

    def __init__(self, AR: int, N: int = 1000):
        """Initialize PhysicsModel.

        Parameters
        ----------
        AR : int
            Aspect ratio of the feature.
        N : int, optional
            Number of spatial grid points. Default 1000.
        """
        self.AR = AR
        self.N = N

    def evaluate(self, xi, **params):
        """Evaluate physics model at specified points and parameters.

        Dispatches to unified continuum solver that handles all parameter
        combinations: s0 (thermal ALD), s_CVD (CVD), r (recombination),
        and d0 (desorption) simultaneously.

        Parameters
        ----------
        xi : array-like
            Spatial evaluation points, shape (M,), values in [0, 1].
        **params : dict
            Parameter dictionary including:
            - 's0': Thermal adsorption sticking probability
            - 'k': Dimensionless dose parameter
            - 'r': (optional) Recombination probability, default 0
            - 'd0': (optional) Desorption probability, default 0
            - 's_CVD': (optional) CVD sticking probability, default 0

        Returns
        -------
        xi : np.ndarray
            Unchanged spatial evaluation points.
        theta : np.ndarray
            Interpolated total thickness profile at xi, shape (M,).

        Examples
        --------
        >>> model = PhysicsModel(AR=10)
        >>> xi, theta = model.evaluate(
        ...     xi=np.array([0.0, 0.5, 1.0]),
        ...     s0=0.8,
        ...     d0=0.01,
        ...     s_CVD=0.05,
        ...     k=50.0
        ... )
        >>> theta.shape
        (3,)
        """
        xi = np.asarray(xi, dtype=float)

        # Use unified model for all cases
        sim = DoseBasedContinuumUnified(
            dim_dose=params["k"],
            sticking_prob=params["s0"],
            # both s0 and k should fail if not applied. We can do withouth the others.
            sticking_prob_cvd=params.get("s_CVD", 0.0),
            recombination_prob=params.get("r", 0.0),
            desorption_prob=params.get("d0", 0.0),
            AR=self.AR,
        )

        sim.solve_transient(target_dose=params["k"], N=self.N)
        return xi, sim.evaluate_at(xi)
