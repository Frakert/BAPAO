"""Likelihood terms for Bayesian parameter inference.

Defines the protocol for likelihood evaluation and provides implementations
for profile-based observations (1D coverage profiles) and scalar Gaussian
observations. Supports parameter transformations and custom predictors.

Key Classes:
  LikelihoodTerm: Protocol for likelihood evaluation
  ProfileObservation: For 1D coverage profile data
  GaussianObservation: For scalar observables
  ScaleParameterTransform: Parameter scaling transformation

Key Functions:
  coerce_likelihood_term: Convert mappings to likelihood terms
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
from scipy.stats import norm

ParameterBatch = dict[str, np.ndarray]
Transform = Callable[[ParameterBatch], Mapping[str, object]]
Predictor = Callable[[object, ParameterBatch], object]


class LikelihoodTerm(Protocol):
    """Protocol for likelihood evaluation in MCMC inference.

    Objects implementing this protocol can be used as likelihood terms in
    Bayesian inference. Requires vectorized evaluation across walker ensemble.
    """

    def log_likelihood_vectorized(self, model, params: ParameterBatch) -> np.ndarray:
        """Compute log-likelihood for parameter ensemble.

        Parameters
        ----------
        model : object
            Surrogate or physics model with evaluate() method.
        params : dict[str, np.ndarray]
            Parameter batch with shape (n_walkers,) per parameter.

        Returns
        -------
        np.ndarray
            Log-likelihood values, shape (n_walkers,).
        """
        ...

    @property
    def num_observations(self) -> int:
        """Number of independent datapoints in observation.

        Returns
        -------
        int
            Count of observations used in likelihood calculation.
        """
        ...

    def summary(self) -> dict[str, object]:
        """Return summary metadata about observation.

        Returns
        -------
        dict[str, object]
            Dictionary with observation type, name, and configuration info.
        """
        ...


def _broadcast_prediction(prediction, observed) -> np.ndarray:
    """Reshape prediction output to match observation broadcasting.

    Handles various prediction shapes from model.evaluate() to ensure
    compatibility with vectorized likelihood computation.

    Parameters
    ----------
    prediction : array-like
        Model prediction, shape varies (scalar, 1D, or 2D).
    observed : array-like
        Observation data for shape reference.

    Returns
    -------
    np.ndarray
        Reshaped prediction, shape (n_walkers, -1).
    """
    observed = np.asarray(observed, dtype=float)
    pred = np.asarray(prediction, dtype=float)

    if observed.ndim == 0:
        observed = observed.reshape(1)

    if pred.ndim == observed.ndim:
        pred = np.atleast_2d(pred)
    elif pred.ndim == observed.ndim + 1:
        pass
    else:
        pred = np.atleast_2d(pred.reshape(pred.shape[0], -1))

    return pred.reshape(pred.shape[0], -1)


def _flatten_observed(value) -> np.ndarray:
    """Flatten observation value to 1D array.

    Handles scalar and array-like observations, converting to flattened float array.

    Parameters
    ----------
    value : array-like
        Observation value (scalar, 1D, or multidimensional).

    Returns
    -------
    np.ndarray
        Flattened 1D array.
    """
    observed = np.asarray(value, dtype=float)
    if observed.ndim == 0:
        observed = observed.reshape(1)
    return observed.reshape(-1)


def _apply_transform(params: ParameterBatch, transform: Transform | None) -> ParameterBatch:
    """Apply optional parameter transformation.

    If transform is provided, applies it to parameters and updates the batch.
    Useful for dose scaling, unit conversions, or derived parameters.

    Parameters
    ----------
    params : dict[str, np.ndarray]
        Parameter batch.
    transform : callable or None
        Transformation function taking params dict, returning updated params dict.

    Returns
    -------
    dict[str, np.ndarray]
        Updated parameter batch (original if no transform).
    """
    updated = dict(params)
    if transform is None:
        return updated

    for name, value in transform(dict(params)).items():
        updated[name] = np.asarray(value, dtype=float)
    return updated


@dataclass
class ProfileObservation(LikelihoodTerm):
    """Coverage profile with automatic early-stop detection + soft censoring."""

    xi_obs: object  # Measured depths only
    theta_obs: object
    theta_sigma: object
    transform: Transform | None = None
    predictor: Predictor | None = None
    name: str | None = None

    # Censoring options
    full_xi: np.ndarray | None = None  # Default set in __post_init__
    censor_value: float = 0.0
    max_censor_points: int = 15  # Limit for performance

    def __post_init__(self):
        self.xi_obs = np.asarray(self.xi_obs, dtype=float)
        self.theta_obs = np.asarray(self.theta_obs, dtype=float)
        self.theta_sigma = np.asarray(self.theta_sigma, dtype=float)

        # Default full trench grid if not provided
        if self.full_xi is None:
            self.full_xi = np.linspace(0.0, 1.0, 200)
        else:
            self.full_xi = np.asarray(self.full_xi, dtype=float)

    @property
    def num_observations(self) -> int:
        return int(np.asarray(self.theta_obs).size)

    @property
    def num_censored(self) -> int:
        censor_d = self._get_censor_depths()
        return len(censor_d) if censor_d is not None else 0

    def _get_censor_depths(self):
        """Automatically detect early stopping and return censor depths."""
        max_meas = np.max(self.xi_obs)
        theta_at_max_depth = self.theta_obs[self.xi_obs == max_meas]

        # check if the depth profile is fully measured (no early stopping)
        if (
            max_meas >= 0.9
            # if the final value is still high, then the censor assumption makes no sense
            or theta_at_max_depth >= 0.1
        ):
            return None  # Full profile

        # Early stopping detected, which means we need to censor the tail of the profile
        tail_mask = self.full_xi > max_meas
        censor_pts = self.full_xi[tail_mask]

        # Subsample to avoid too many points
        if len(censor_pts) > self.max_censor_points:
            idx = np.linspace(0, len(censor_pts) - 1, self.max_censor_points, dtype=int)
            censor_pts = censor_pts[idx]

        return censor_pts

    def log_likelihood_vectorized(self, model, params):
        obs_params = _apply_transform(params, self.transform)

        # Measured points likelihood
        if self.predictor is None:
            _, pred_meas = model.evaluate(self.xi_obs, **obs_params)
        else:
            pred_meas = self.predictor(model, obs_params)

        pred_meas = _broadcast_prediction(pred_meas, self.theta_obs)

        observed = _flatten_observed(self.theta_obs)
        sigma = _flatten_observed(self.theta_sigma)
        sigma2 = sigma**2

        resid = pred_meas - observed.reshape(1, -1)
        logl = -0.5 * np.sum((resid**2 / sigma2) + np.log(2 * np.pi * sigma2), axis=1)

        # Soft censoring
        censor_depths = self._get_censor_depths()
        if censor_depths is not None and len(censor_depths) > 0:
            if self.predictor is None:
                _, pred_censor = model.evaluate(censor_depths, **obs_params)
            else:
                pred_censor = self.predictor(model, obs_params)

            pred_censor = _broadcast_prediction(pred_censor, censor_depths)
            sigma_c = sigma[0] if np.isscalar(sigma) else sigma[-1]

            z = (self.censor_value - pred_censor) / sigma_c
            logl += np.sum(norm.logcdf(z), axis=1)

        return logl


@dataclass
class GaussianObservation:
    """Scalar Gaussian likelihood observation.

    Represents a single or multiple scalar observables with Gaussian error model.
    Requires explicit predictor function (unlike ProfileObservation).

    Attributes
    ----------
    observed : float or np.ndarray
        Observed value(s).
    sigma : float or np.ndarray
        Noise standard deviation(s).
    predictor : callable
        Function computing prediction from model: predictor(model, params) -> value.
    transform : callable, optional
        Parameter transformation applied before prediction.
    name : str, optional
        Name/label for observation.

    Examples
    --------
    >>> obs = GaussianObservation(
    ...     observed=5.0,
    ...     sigma=0.1,
    ...     predictor=lambda model, params: params['s0'] * params['k'],
    ...     name='growth_rate'
    ... )
    """

    observed: object
    sigma: object
    predictor: Predictor
    transform: Transform | None = None
    name: str | None = None

    def __post_init__(self):
        self.observed = np.asarray(self.observed, dtype=float)
        self.sigma = np.asarray(self.sigma, dtype=float)

    @property
    def num_observations(self) -> int:
        return int(np.asarray(self.observed).size)

    def log_likelihood_vectorized(self, model, params: ParameterBatch) -> np.ndarray:
        obs_params = _apply_transform(params, self.transform)
        prediction = self.predictor(model, obs_params)
        prediction = _broadcast_prediction(prediction, self.observed)
        observed = _flatten_observed(self.observed)
        sigma2 = _flatten_observed(self.sigma) ** 2
        resid = prediction - observed.reshape(1, -1)
        return -0.5 * np.sum((resid**2 / sigma2) + np.log(2 * np.pi * sigma2), axis=1)

    def summary(self) -> dict[str, object]:
        return {
            "kind": "gaussian",
            "name": self.name or "gaussian",
            "num_observations": self.num_observations,
            "has_transform": self.transform is not None,
            "transform": type(self.transform).__name__ if self.transform is not None else None,
            "uses_custom_predictor": True,
        }


@dataclass(frozen=True)
class ScaleParameterTransform:
    """Parameter scaling transformation for multi-condition observations.

    Applies a scalar factor to a parameter (e.g., dose scaling in paired profiles).
    Useful for linking observations that share parameters but at different scales.

    Attributes
    ----------
    parameter : str
        Parameter name to scale (e.g., 'k' for dose).
    factor : float
        Multiplicative factor to apply.

    Examples
    --------
    >>> transform = ScaleParameterTransform(parameter='k', factor=2.0)
    >>> result = transform({'k': np.array([100.0]), 's0': np.array([0.5])})
    >>> result  # doctest: +SKIP
    {'k': array([200.])}
    """

    parameter: str
    factor: float

    def __call__(self, params: ParameterBatch) -> Mapping[str, object]:
        """Apply scaling to specified parameter.

        Parameters
        ----------
        params : dict[str, np.ndarray]
            Parameter batch.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary with scaled parameter value only.
        """
        return {self.parameter: params[self.parameter] * self.factor}


def coerce_likelihood_term(observation) -> LikelihoodTerm:
    """Convert observation mapping or object to LikelihoodTerm.

    Accepts either objects implementing the LikelihoodTerm protocol or
    mappings with profile observation keys, converting the latter to
    ProfileObservation instances.

    Parameters
    ----------
    observation : LikelihoodTerm, dict, or Mapping
        Either a LikelihoodTerm-compatible object or mapping with keys:
        'xi_obs', 'theta_obs', 'theta_sigma' (required), plus optionally
        'transform', 'predictor', 'name'.

    Returns
    -------
    LikelihoodTerm
        Likelihood term object ready for MCMC inference.

    Raises
    ------
    TypeError
        If observation type not recognized.
    ValueError
        If observation mapping missing required keys.

    Examples
    --------
    >>> obs_dict = {
    ...     'xi_obs': np.linspace(0, 1, 50),
    ...     'theta_obs': np.ones(50) * 0.5,
    ...     'theta_sigma': 0.05
    ... }
    >>> term = coerce_likelihood_term(obs_dict)  # doctest: +SKIP
    """
    if hasattr(observation, "log_likelihood_vectorized") and hasattr(
        observation, "num_observations"
    ):
        return observation

    if not isinstance(observation, Mapping):
        raise TypeError(
            "Observations must be mappings or likelihood-term objects with "
            "'log_likelihood_vectorized' and 'num_observations'."
        )

    if {"xi_obs", "theta_obs", "theta_sigma"}.issubset(observation):
        return ProfileObservation(
            xi_obs=observation["xi_obs"],
            theta_obs=observation["theta_obs"],
            theta_sigma=observation["theta_sigma"],
            transform=observation.get("transform"),
            predictor=observation.get("predictor"),
            name=observation.get("name"),
        )

    raise ValueError("Unsupported observation mapping. Expected profile observation keys.")
