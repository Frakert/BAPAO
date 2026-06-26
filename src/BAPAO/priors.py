"""
Prior terms for Bayesian parameter inference.

Defines the protocol for prior evaluation and provides implementations for

    * UniformPrior
    * GaussianPrior
    * DoubleExpPrior

Each prior evaluates a vectorized log-prior contribution for an ensemble of
walkers. Parameter bounds are intentionally NOT checked here; they remain the
responsibility of the MCMC engine. This module only contributes informative
prior terms.

Key Classes
-----------
PriorTerm
    Protocol for prior evaluation.

UniformPrior
    Flat prior (returns zero everywhere inside the allowed parameter region).

GaussianPrior
    Independent Gaussian prior on one or more parameters.

DoubleExpPrior
    Physics-informed prior based on the double-exponential fit from the paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

ParameterBatch = dict[str, np.ndarray]


class PriorTerm(Protocol):
    """Protocol for vectorized prior evaluation."""

    def log_prior_vectorized(
        self,
        params: ParameterBatch,
    ) -> np.ndarray: ...

    def summary(self) -> dict[str, object]: ...


# ---------------------------------------------------------------------------
# Uniform prior
# ---------------------------------------------------------------------------


@dataclass
class UniformPrior:
    """Flat prior.

    Returns zero for every walker. Parameter bounds are enforced separately
    by the MCMC engine.
    """

    def log_prior_vectorized(
        self,
        params: ParameterBatch,
    ) -> np.ndarray:

        n = next(iter(params.values())).shape[0]
        return np.zeros(n)

    def summary(self) -> dict[str, object]:

        return {
            "kind": "uniform",
        }


# ---------------------------------------------------------------------------
# Gaussian prior
# ---------------------------------------------------------------------------

@dataclass
class GaussianPrior:
    """Independent Gaussian prior.

    Parameters
    ----------
    means
        Dictionary mapping parameter name -> prior mean.

    sigmas
        Dictionary mapping parameter name -> prior standard deviation.

    Notes
    -----
    Only parameters listed in ``means`` are constrained.

    Example
    -------
    >>> prior = GaussianPrior(
    ...     means={"s0": 0.02},
    ...     sigmas={"s0": 0.005},
    ... )
    """

    means: dict[str, float]
    sigmas: dict[str, float]

    def __post_init__(self):

        if set(self.means) != set(self.sigmas):
            raise ValueError("'means' and 'sigmas' must contain identical parameter names.")

        for sigma in self.sigmas.values():
            if sigma <= 0:
                raise ValueError("Gaussian prior sigma must be positive.")

    def log_prior_vectorized(
        self,
        params: ParameterBatch,
    ) -> np.ndarray:

        n = next(iter(params.values())).shape[0]

        lp = np.zeros(n)

        for name in self.means:
            if name not in params:
                continue

            mean = self.means[name]
            sigma = self.sigmas[name]

            x = params[name]

            lp += -0.5 * (((x - mean) / sigma) ** 2 + np.log(2.0 * np.pi * sigma**2))

        return lp

    def summary(self) -> dict[str, object]:

        return {
            "kind": "gaussian",
            "means": self.means,
            "sigmas": self.sigmas,
        }

# ---------------------------------------------------------------------------
# Double exponential prior
# --------------------------------------------------------------------------

@dataclass
class DoubleExpPrior:
    """Physics-informed prior derived from the double-exponential fit.

    Implements Eq. (7) of the paper

        ln p(s0,r)
            = -((0.79*s0 + r - D)^2)/(2*sigma_D^2)

    where

        D = (4/3) c² AR⁻²

    The uniform part of the prior is handled separately by the parameter
    bounds in the MCMC engine.
    """

    c: float

    aspect_ratio: float = 1000.0

    sigma_D: float | None = None

    def _D(self) -> float:

        return (4.0 / 3.0) * self.c**2 * self.aspect_ratio**-2

    def log_prior_vectorized(
        self,
        params: ParameterBatch,
    ) -> np.ndarray:

        s0 = params.get("s0")
        r = params.get("r")

        if s0 is None or r is None:
            n = next(iter(params.values())).shape[0]
            return np.zeros(n)

        D = self._D()

        sigma = self.sigma_D

        if sigma is None:
            sigma = 0.5 * D

        combination = 0.79 * s0 + r

        return -((combination - D) ** 2) / (2.0 * sigma**2)

    def summary(self) -> dict[str, object]:

        return {
            "kind": "double_exp",
            "c": self.c,
            "aspect_ratio": self.aspect_ratio,
            "D": self._D(),
            "sigma_D": self.sigma_D,
        }


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def coerce_prior_term(prior) -> PriorTerm:
    """Convert an object into a PriorTerm."""
    if prior is None:
        return UniformPrior()

    if hasattr(prior, "log_prior_vectorized"):
        return prior

    raise TypeError("Prior must implement 'log_prior_vectorized()'.")
