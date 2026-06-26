"""Preset configurations and observation factories for common ALD scenarios.

Provides helper functions to quickly construct MCMC configurations and
likelihood terms for standard ALD processes (thermal, desorption, CVD).
Encapsulates parameter bounds and constraints for reproducibility.

Key Functions:
  make_config: Create MCMC configuration with parameters and bounds
  thermal_config: Preset for basic thermal ALD (s0, k)
  desorption_config: Preset for thermal with desorption (s0, k, d0)
  plasma_config: Preset for plasma with recombination (s0, k, r)
  single_profile_observation: Create observation from profile data
  paired_profile_observations: Create paired observations at different doses
  gaussian_observation: Create scalar Gaussian likelihood term
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from BAPAO.constraints import constraint_s0_gt_d0
from BAPAO.likelihoods import GaussianObservation, ProfileObservation, ScaleParameterTransform
from BAPAO.parameters import validate_config


def make_config(
    *,
    names: Sequence[str],
    bounds: Mapping[str, tuple[float, float]],
    fixed_params: Mapping[str, float] | None = None,
    constraint=None,
) -> dict[str, object]:
    """Create MCMC configuration for parameter inference.

    Assembles parameter names, bounds, fixed parameters, and optional
    constraint function into a validated configuration dictionary.

    Parameters
    ----------
    names : Sequence[str]
        Active parameter names (e.g., ['s0', 'k']).
    bounds : Mapping[str, tuple[float, float]]
        Parameter name to (lower, upper) bound pairs (linear space).
    fixed_params : Mapping[str, float], optional
        Fixed parameter values not varied during inference.
    constraint : callable, optional
        Constraint function taking param dict and returning bool.
        Applied to reject samples violating physical constraints.

    Returns
    -------
    dict[str, object]
        Validated configuration dict with 'names', 'bounds', 'fixed_params',
        and optionally 'constraint'.

    Raises
    ------
    ValueError
        If config fails validation (e.g., missing bounds).

    Examples
    --------
    >>> config = make_config(
    ...     names=['d0', 's0'],
    ...     bounds={'d0': (1e-8, 1e-1), 's0': (1e-5, 1e-1)},
    ...     fixed_params={'k': 100.0}
    ... )
    >>> config['names']
    ['d0', 's0']
    """
    config: dict[str, object] = {
        "names": list(names),
        "bounds": dict(bounds),
        "fixed_params": dict(fixed_params or {}),
    }
    if constraint is not None:
        config["constraint"] = constraint
    return validate_config(config)


def single_profile_observation(xi_obs, theta_obs, theta_sigma):
    """Create a single profile observation for MCMC likelihood.

    Wraps coverage profile data into a ProfileObservation object for
    use as a likelihood term in MCMC inference.

    Parameters
    ----------
    xi_obs : array-like
        Normalized spatial points (depth), shape (N,), values in [0, 1].
    theta_obs : array-like
        Observed normalized coverage/thickness, shape (N,), values in [0, 1].
    theta_sigma : float or array-like
        Observation noise standard deviation. If scalar, applied uniformly.
        If array, must match theta_obs shape.

    Returns
    -------
    ProfileObservation
        Observation object suitable for likelihood evaluation.

    Examples
    --------
    >>> import numpy as np
    >>> xi = np.linspace(0, 1, 50)
    >>> theta = 0.5 * xi
    >>> obs = single_profile_observation(xi, theta, 0.05)
    >>> obs.xi_obs.shape
    (50,)
    """
    return ProfileObservation(xi_obs=xi_obs, theta_obs=theta_obs, theta_sigma=theta_sigma)


def gaussian_observation(observed, sigma, predictor, *, transform=None, name: str | None = None):
    """Create a scalar Gaussian likelihood observation.

    Wraps a scalar measurement and predictor function into a GaussianObservation
    for likelihood evaluation. Used for non-profile observables.

    Parameters
    ----------
    observed : float
        Observed value.
    sigma : float
        Measurement noise standard deviation.
    predictor : callable
        Function that predicts the observable given parameters.
        Called as predictor(**param_dict).
    transform : callable, optional
        Transformation function applied to parameters before prediction.
    name : str, optional
        Name/label for this observation.

    Returns
    -------
    GaussianObservation
        Observation object for likelihood evaluation.

    Examples
    --------
    >>> obs = gaussian_observation(
    ...     observed=0.5,
    ...     sigma=0.05,
    ...     predictor=lambda x: x**2,
    ...     name='quadratic_measurement'
    ... )
    """
    return GaussianObservation(
        observed=observed,
        sigma=sigma,
        predictor=predictor,
        transform=transform,
        name=name,
    )


def paired_profile_observations(
    *,
    xi_obs,
    profiles_obs,
    sigma_profiles,
    dose_ratio: float,
):
    """Create paired profile observations at different doses.

    Generates two ProfileObservation objects for profiles measured at
    different doses, with the second observation linked to the first via
    a dose scaling transform on the 'k' parameter.

    Parameters
    ----------
    xi_obs : array-like or tuple of array-like
        Spatial points for observation(s). If single array, applied to both
        profiles. If tuple, (xi_1, xi_2) for each profile.
    profiles_obs : sequence of array-like
        Two coverage profiles, [theta_obs_1, theta_obs_2].
    sigma_profiles : float or tuple of float
        Noise level(s). If scalar, applied to both. If tuple, (sigma_1, sigma_2).
    dose_ratio : float
        Dose ratio k_2 / k_1 applied as transform in second observation.

    Returns
    -------
    list[ProfileObservation]
        List of two ProfileObservation objects. Second has ScaleParameterTransform.

    Examples
    --------
    >>> xi = np.linspace(0, 1, 50)
    >>> theta1 = 0.5 * xi
    >>> theta2 = 0.7 * xi
    >>> obs_list = paired_profile_observations(
    ...     xi_obs=xi,
    ...     profiles_obs=[theta1, theta2],
    ...     sigma_profiles=0.05,
    ...     dose_ratio=2.0
    ... )
    >>> len(obs_list)
    2
    """
    if isinstance(sigma_profiles, (list, tuple)):
        sigma1, sigma2 = sigma_profiles
    else:
        sigma1 = sigma2 = sigma_profiles

    if isinstance(xi_obs, (list, tuple)):
        xi_1, xi_2 = xi_obs
    else:
        xi_1 = xi_2 = xi_obs

    return [
        single_profile_observation(xi_1, profiles_obs[0], sigma1),
        ProfileObservation(
            xi_obs=xi_2,
            theta_obs=profiles_obs[1],
            theta_sigma=sigma2,
            transform=ScaleParameterTransform(parameter="k", factor=dose_ratio),
            name="paired_profile_scaled_dose",
        ),
    ]


def thermal_config(bounds: Mapping[str, tuple[float, float]]):
    """Create configuration for basic thermal ALD (s0, k only).

    Convenience function for thermal ALD inference without desorption
    or recombination effects. Parameters: sticking (s0) and dose (k).

    Parameters
    ----------
    bounds : Mapping[str, tuple[float, float]]
        Parameter bounds for 's0' and 'k' (e.g., s0: [1e-5, 1e-1]).

    Returns
    -------
    dict[str, object]
        Validated config for (s0, k) parameters.

    Examples
    --------
    >>> bounds = {'s0': (1e-5, 1e-1), 'k': (1e-5, 1e1)}
    >>> config = thermal_config(bounds)
    >>> config['names']
    ['s0', 'k']
    """
    return make_config(names=["s0", "k"], bounds=bounds)


def desorption_config(bounds: Mapping[str, tuple[float, float]]):
    """Create configuration for thermal ALD with desorption (s0, k, d0).

    Configuration for desorption-limited thermal ALD with constraint
    that sticking (s0) > desorption (d0) for physical realism.

    Parameters
    ----------
    bounds : Mapping[str, tuple[float, float]]
        Parameter bounds for 's0', 'k', and 'd0'. Typically s0 > d0.

    Returns
    -------
    dict[str, object]
        Validated config for (s0, k, d0) with constraint_s0_gt_d0.

    Examples
    --------
    >>> bounds = {'s0': (1e-5, 1e-1), 'k': (1e-5, 1e1), 'd0': (1e-8, 1e-1)}
    >>> config = desorption_config(bounds)
    >>> config['names']
    ['s0', 'k', 'd0']
    """
    return make_config(
        names=["s0", "k", "d0"],
        bounds=bounds,
        constraint=constraint_s0_gt_d0,
    )


def plasma_config(bounds: Mapping[str, tuple[float, float]]):
    """Create configuration for plasma ALD with recombination (s0, k, r).

    Configuration for plasma/ion-assisted ALD with recombination losses.
    Parameters: sticking (s0), dose (k), recombination (r).

    Parameters
    ----------
    bounds : Mapping[str, tuple[float, float]]
        Parameter bounds for 's0', 'k', and 'r'.

    Returns
    -------
    dict[str, object]
        Validated config for (s0, k, r) parameters.

    Examples
    --------
    >>> bounds = {'s0': (1e-5, 1e-1), 'k': (1e-5, 1e1), 'r': (1e-20, 1e-5)}
    >>> config = plasma_config(bounds)
    >>> config['names']
    ['s0', 'k', 'r']
    """
    return make_config(names=["s0", "k", "r"], bounds=bounds)
