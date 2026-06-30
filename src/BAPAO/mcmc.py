"""MCMC inference engine for Bayesian parameter estimation.

This module provides vectorized MCMC sampling using emcee's ensemble sampler.
It computes prior/likelihood/posterior in log-space, handles fixed parameters,
applies physical constraints, and post-processes samples for parameter estimation.

Key Functions:
  run_mcmc: Main entry point for MCMC inference
  log_prior_vectorized: Log-prior evaluation across walker ensemble
  log_likelihood_vectorized: Log-likelihood evaluation for observations
  log_posterior_vectorized: Combined log-posterior (prior + likelihood)
  compute_bic: Bayesian Information Criterion for model comparison
  observation_count: Total number of observation datapoints
  DoubleExpPrior: Physics-informed prior derived from the double-exponential fit
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence

import numpy as np

from BAPAO.likelihoods import LikelihoodTerm, coerce_likelihood_term
from BAPAO.parameters import resolve_config_param_names, validate_config
from BAPAO.priors import PriorTerm

# ---------------------------------------------------------------------------
# Parameter unpacking helpers
# ---------------------------------------------------------------------------


def unpack_theta(theta_batch: np.ndarray, param_names: Sequence[str]) -> dict[str, np.ndarray]:
    """Convert log10-space parameter batch to linear space.

    Parameters
    ----------
    theta_batch : np.ndarray
        Shape (n_walkers, n_params). Each column is a parameter in log10 space.
    param_names : Sequence[str]
        Names of parameters corresponding to columns of theta_batch.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary mapping parameter names to (n_walkers,) arrays in linear space.
    """
    return {name: 10 ** theta_batch[:, index] for index, name in enumerate(param_names)}


def resolve_parameter_batch(
    theta_batch: np.ndarray, config: Mapping[str, object]
) -> dict[str, np.ndarray]:
    """Unpack and combine active and fixed parameters into a resolved batch.

    Parameters
    ----------
    theta_batch : np.ndarray
        Shape (n_walkers, n_active_params). Active parameters in log10 space.
    config : Mapping[str, object]
        Configuration dict with 'names', 'fixed_params', and 'bounds'.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary mapping all parameter names (active + fixed) to (n_walkers,) arrays.

    Raises
    ------
    ValueError
        If fixed parameter has length != 1 and != n_walkers.
    """
    config = validate_config(config)
    param_names = config["names"]  # type: ignore[assignment]
    fixed_params = config.get("fixed_params", {})  # type: ignore[assignment]

    resolved = unpack_theta(theta_batch, param_names)
    n_walkers = theta_batch.shape[0]

    for name, value in fixed_params.items():
        array = np.atleast_1d(np.asarray(value, dtype=float))
        if array.size == 1 and n_walkers > 1:
            array = np.repeat(array, n_walkers)
        elif array.size != n_walkers:
            raise ValueError(
                f"Fixed parameter '{name}' has length {array.size}, expected 1 or {n_walkers}."
            )
        resolved[name] = array

    return resolved


# ---------------------------------------------------------------------------
# Log-prior
# ---------------------------------------------------------------------------


def log_prior_vectorized(
    theta_batch: np.ndarray,
    config: Mapping[str, object],
    prior: PriorTerm | None = None,
) -> np.ndarray:
    """Vectorized log-prior evaluation across walker ensemble.

    Combines a uniform (flat) prior over the parameter bounds with an optional
    physics-informed Gaussian prior derived from the double-exponential fit
    (Eq. 7 in the paper). The Gaussian term only has effect when both 's0'
    and 'r' are active parameters (the PEALD case).

    Parameters
    ----------
    theta_batch : np.ndarray
        Shape (n_walkers, n_params). Parameters in log10 space.
    config : Mapping[str, object]
        Configuration dict with 'names', 'bounds', and optional 'constraint'.
    double_exp_prior : DoubleExpPrior or None, optional
        Physics-informed prior instance. Pass ``None`` (default) to use a
        purely flat prior (original behaviour).

    Returns
    -------
    np.ndarray
        Shape (n_walkers,). Log-prior values (0 if valid and no informative
        prior; -inf if outside bounds or constraint violated; <= 0 otherwise).
    """
    param_names = config["names"]  # type: ignore[assignment]
    bounds = config["bounds"]  # type: ignore[assignment]

    n = theta_batch.shape[0]
    lp = np.zeros(n)
    valid = np.ones(n, dtype=bool)

    # --- Uniform (flat) prior: zero inside bounds, -inf outside --------------
    for index, name in enumerate(param_names):
        lo, hi = bounds[name]  # type: ignore[index]
        log_lo, log_hi = np.log10(lo), np.log10(hi)
        valid &= (theta_batch[:, index] >= log_lo) & (theta_batch[:, index] <= log_hi)

    # --- Optional physical constraint ----------------------------------------
    constraint = config.get("constraint")
    if constraint is not None:
        params = resolve_parameter_batch(theta_batch, config)
        all_param_names = resolve_config_param_names(config)
        physical = np.array(
            [
                bool(constraint({name: params[name][row] for name in all_param_names}))
                for row in range(n)
            ]
        )
        valid &= physical

    lp[~valid] = -np.inf

    if prior is not None and np.any(valid):
        params_valid = resolve_parameter_batch(theta_batch[valid], config)
        lp[valid] += prior.log_prior_vectorized(params_valid)

    return lp


# ---------------------------------------------------------------------------
# Observation normalisation helpers
# ---------------------------------------------------------------------------


def _normalize_observation(observation: Mapping[str, object]) -> dict[str, object]:
    normalized = dict(observation)
    normalized["xi_obs"] = np.asarray(observation["xi_obs"], dtype=float)
    normalized["theta_obs"] = np.asarray(observation["theta_obs"], dtype=float)
    normalized["theta_sigma"] = np.asarray(observation["theta_sigma"], dtype=float)
    return normalized


def _normalize_observations(
    *,
    observations: Sequence[object] | None = None,
    xi_obs=None,
    theta_obs=None,
    theta_sigma=None,
) -> list[LikelihoodTerm]:
    if observations is None:
        if xi_obs is None or theta_obs is None or theta_sigma is None:
            raise ValueError(
                "Provide either 'observations' or the trio 'xi_obs', 'theta_obs', 'theta_sigma'."
            )
        observations = [
            {
                "xi_obs": xi_obs,
                "theta_obs": theta_obs,
                "theta_sigma": theta_sigma,
            }
        ]

    return [coerce_likelihood_term(observation) for observation in observations]


# ---------------------------------------------------------------------------
# Log-likelihood and log-posterior
# ---------------------------------------------------------------------------


def log_likelihood_vectorized(
    theta_batch: np.ndarray,
    model,
    observations: Sequence[LikelihoodTerm],
    config: Mapping[str, object],
) -> np.ndarray:
    """Vectorized log-likelihood evaluation across walker ensemble.

    Parameters
    ----------
    theta_batch : np.ndarray
        Shape (n_walkers, n_params). Parameters in log10 space.
    model : SurrogateModel
        Surrogate model for computing predictions.
    observations : Sequence[LikelihoodTerm]
        List of likelihood terms (profile observations, gaussian observations, etc).
    config : Mapping[str, object]
        Configuration dict with active and fixed parameters.

    Returns
    -------
    np.ndarray
        Shape (n_walkers,). Sum of log-likelihoods from all observations.
    """
    params = resolve_parameter_batch(theta_batch, config)
    n_walkers = theta_batch.shape[0]
    total_log_like = np.zeros(n_walkers)

    for observation in observations:
        total_log_like += observation.log_likelihood_vectorized(model, params)

    return total_log_like


def log_posterior_vectorized(
    theta_batch: np.ndarray,
    config: Mapping[str, object],
    model,
    observations: Sequence[LikelihoodTerm],
    prior: PriorTerm | None = None,
) -> np.ndarray:
    """Vectorized log-posterior evaluation (log-prior + log-likelihood).

    Parameters
    ----------
    theta_batch : np.ndarray
        Shape (n_walkers, n_params). Parameters in log10 space.
    config : Mapping[str, object]
        Configuration dict with 'names', 'bounds', fixed/active parameters.
    model : SurrogateModel
        Surrogate model for likelihood computation.
    observations : Sequence[LikelihoodTerm]
        Likelihood terms for observation data.
    prior : PriorTerm or None, optional
        Prior term for model parameters. Forwarded to log_prior_vectorized.

    Returns
    -------
    np.ndarray
        Shape (n_walkers,). Log-posterior values (prior + likelihood for valid
        samples, -inf otherwise).
    """
    lp = log_prior_vectorized(theta_batch, config, prior=prior)

    valid = np.isfinite(lp)
    if np.any(valid):
        lp[valid] += log_likelihood_vectorized(theta_batch[valid], model, observations, config)

    return lp


# ---------------------------------------------------------------------------
# Miscellaneous helpers
# ---------------------------------------------------------------------------


def _default_log_center(bounds: tuple[float, float]) -> float:
    return 0.5 * (np.log10(bounds[0]) + np.log10(bounds[1]))


def observation_count(observations: Sequence[LikelihoodTerm]) -> int:
    """Sum the total number of observation datapoints across all likelihood terms.

    Parameters
    ----------
    observations : Sequence[LikelihoodTerm]
        List of likelihood terms.

    Returns
    -------
    int
        Total number of observation datapoints.
    """
    return int(sum(observation.num_observations for observation in observations))


def _default_postprocess_batch_size(observations: Sequence[LikelihoodTerm]) -> int:
    largest_observation = max(
        (int(observation.num_observations) for observation in observations), default=1
    )
    target_model_rows = 131_072
    return max(32, target_model_rows // max(1, largest_observation))


def _batched_log_likelihood(
    theta_batch: np.ndarray,
    model,
    observations: Sequence[LikelihoodTerm],
    config: Mapping[str, object],
    *,
    batch_size: int,
) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    chunks = []
    for start in range(0, theta_batch.shape[0], batch_size):
        stop = min(start + batch_size, theta_batch.shape[0])
        chunks.append(
            log_likelihood_vectorized(theta_batch[start:stop], model, observations, config)
        )
    return np.concatenate(chunks, axis=0) if chunks else np.empty(0, dtype=float)


def compute_bic(max_log_likelihood: float, n_parameters: int, n_observations: int) -> float | None:
    """Compute Bayesian Information Criterion for model comparison.

    BIC = k*ln(n) - 2*ln(L_max), where k=n_parameters, n=n_observations, L_max=max likelihood.
    Lower BIC indicates better fit (penalizing model complexity).

    Parameters
    ----------
    max_log_likelihood : float
        Maximum log-likelihood value found during inference.
    n_parameters : int
        Number of model parameters.
    n_observations : int
        Number of observation datapoints.

    Returns
    -------
    float | None
        BIC value, or None if n_observations <= 0 or likelihood is not finite.
    """
    if n_observations <= 0 or not np.isfinite(max_log_likelihood):
        return None
    return float(n_parameters * np.log(n_observations) - 2.0 * max_log_likelihood)


def _serialize_config(config: Mapping[str, object]) -> dict[str, object]:
    serialized = dict(config)
    if "constraint" in serialized:
        serialized["constraint"] = repr(serialized["constraint"])
    return serialized


def _tracker_params(
    config: Mapping[str, object],
    *,
    nwalkers: int,
    nsteps: int,
    burn_frac: float,
    postprocess_batch_size: int,
    observations: Sequence[LikelihoodTerm],
    prior: PriorTerm | None,
) -> dict[str, object]:
    params = {
        "active_parameters": ",".join(config["names"]),  # type: ignore[index]
        "fixed_parameters": ",".join(sorted(config.get("fixed_params", {}).keys())),
        "nwalkers": nwalkers,
        "nsteps": nsteps,
        "burn_frac": burn_frac,
        "postprocess_batch_size": postprocess_batch_size,
        "n_parameters": len(config["names"]),  # type: ignore[arg-type]
        "n_observations": observation_count(observations),
        "prior": prior.summary() if prior is not None else {"kind": "uniform"},
    }

    return params


def _build_initial_positions(
    config: Mapping[str, object],
    nwalkers: int,
    initial_estimates: Mapping[str, float] | None = None,
    initial_spreads: Mapping[str, float] | None = None,
    random_state: np.random.Generator | None = None,
    prior: PriorTerm | None = None,
) -> np.ndarray:
    rng = random_state or np.random.default_rng()
    config = validate_config(config)
    param_names = config["names"]  # type: ignore[assignment]
    bounds = config["bounds"]  # type: ignore[assignment]

    ndim = len(param_names)
    p0 = np.zeros((nwalkers, ndim))

    for index, name in enumerate(param_names):
        lo, hi = bounds[name]  # type: ignore[index]
        center_value = (
            initial_estimates[name]
            if initial_estimates is not None and name in initial_estimates
            else np.sqrt(lo * hi)
        )
        spread = (
            initial_spreads[name]
            if initial_spreads is not None and name in initial_spreads
            else 0.5
        )
        center = np.log10(center_value)
        p0[:, index] = rng.normal(center, spread, nwalkers)
        p0[:, index] = np.clip(p0[:, index], np.log10(lo), np.log10(hi))

    for _ in range(100):
        lp = log_prior_vectorized(p0, config, prior=prior)
        invalid = ~np.isfinite(lp)
        if not np.any(invalid):
            return p0

        for row in np.where(invalid)[0]:
            for index, name in enumerate(param_names):
                lo, hi = bounds[name]  # type: ignore[index]
                center = (
                    np.log10(initial_estimates[name])
                    if initial_estimates is not None and name in initial_estimates
                    else _default_log_center(bounds[name])  # type: ignore[index]
                )
                spread = (
                    initial_spreads[name]
                    if initial_spreads is not None and name in initial_spreads
                    else 0.5
                )
                p0[row, index] = np.clip(
                    rng.normal(center, spread),
                    np.log10(lo),
                    np.log10(hi),
                )

    raise RuntimeError("Failed to generate valid initial walker positions for the supplied config.")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_mcmc(
    *,
    model,
    config: Mapping[str, object],
    observations: Sequence[Mapping[str, object]] | None = None,
    xi_obs=None,
    theta_obs=None,
    theta_sigma=None,
    nwalkers: int = 128,
    nsteps: int = 10_000,
    burn_frac: float = 0.25,
    initial_estimates: Mapping[str, float] | None = None,
    initial_spreads: Mapping[str, float] | None = None,
    init_theta_log: np.ndarray | None = None,
    random_state: np.random.Generator | None = None,
    postprocess_batch_size: int | None = None,
    prior: PriorTerm | None = None,
):
    """Run ensemble MCMC inference with emcee.

    Performs affine-invariant ensemble sampling with automatic thinning, computes
    posterior samples in linear space, and returns MAP/MLE estimates, BIC, and
    full sample chains.

    Parameters
    ----------
    model : SurrogateModel
        Neural network surrogate model for predictions.
    config : Mapping[str, object]
        Configuration dict with 'names' (active params), 'bounds', and optional
        'fixed_params'. Built via make_config(), thermal_config(), etc.
    observations : Sequence[Mapping[str, object]], optional
        List of observation dicts (ProfileObservation, GaussianObservation, etc).
        If None, must provide xi_obs, theta_obs, theta_sigma.
    xi_obs : array-like, optional
        Observation points (profile position/dose). Used if observations=None.
    theta_obs : array-like, optional
        Observed values at xi_obs. Used if observations=None.
    theta_sigma : float or array-like, optional
        Gaussian noise standard deviation. Used if observations=None.
    nwalkers : int, optional
        Number of MCMC walkers (must be even). Default: 128.
    nsteps : int, optional
        Number of MCMC steps per walker. Default: 10000.
    burn_frac : float, optional
        Fraction of steps to discard (burn-in). Default: 0.25.
    initial_estimates : Mapping[str, float], optional
        Starting point estimates for parameters (linear space).
    initial_spreads : Mapping[str, float], optional
        Initial position spreads in log10 space. Default: 0.5.
    init_theta_log : np.ndarray, optional
        Pre-computed initial walker positions (n_walkers, n_params) in log10 space.
    random_state : np.random.Generator, optional
        Random number generator for reproducibility.
    postprocess_batch_size : int, optional
        Batch size for likelihood post-processing. Default: auto-computed.
    prior :Prior or None, optional
        Physics-informed Gaussian prior built from a double-exponential fit
        (Eq. 7 of the paper). Constrains (0.79*s0 + r) near D = (4/3)*c^2*AR^-2.
        Only has effect when both 's0' and 'r' are active parameters (PEALD).
        For thermal ALD the prior silently returns 0 (flat). Construct via::

            fitter = FitModule()
            popt, _ = fitter.fit_profile(xi_obs, theta_obs,
                                          model_type=ModelType.DOUBLE_EXP_STABLE,
                                          ald_type=ALDType.THERMAL)
            prior = Prior(c=popt[1], aspect_ratio=1000)

        Default: None (flat uniform prior, original behaviour).

    Returns
    -------
    dict
        Results dict with keys:
          - 'samples': (n_samples, n_params) posterior samples in linear space
          - 'samples_log': (n_samples, n_params) posterior samples in log10 space
          - 'logp': (n_samples,) log-posterior values
          - 'loglike': (n_samples,) log-likelihood values
          - 'parameter_names': names of sampled parameters (active only)
          - 'all_parameter_names': names of all parameters (active + fixed)
          - 'map': (n_params,) maximum-a-posteriori estimate in linear space
          - 'mle': (n_params,) maximum-likelihood estimate in linear space
          - 'max_loglike': highest log-likelihood value
          - 'bic': Bayesian Information Criterion for model comparison
          - 'n_parameters': number of active parameters
          - 'n_observations': total number of data points
          - 'acceptance_fraction': fraction of accepted MCMC moves
          - 'fixed_params': dict of fixed parameter values
          - 'runtime_seconds': wall-clock inference time
          - 'sampler': emcee.EnsembleSampler instance (full chain history)
          - 'thin': thinning factor applied during post-processing
          - 'prior': the DoubleExpPrior used (or None)
    """
    import emcee

    config = validate_config(config)
    observations_list = _normalize_observations(
        observations=observations,
        xi_obs=xi_obs,
        theta_obs=theta_obs,
        theta_sigma=theta_sigma,
    )

    if postprocess_batch_size is None:
        postprocess_batch_size = _default_postprocess_batch_size(observations_list)

    ndim = len(config["names"])  # type: ignore[arg-type]
    if ndim == 0:
        raise ValueError("At least one parameter must remain free for inference.")
    if init_theta_log is None:
        p0 = _build_initial_positions(
            config,
            nwalkers,
            initial_estimates=initial_estimates,
            initial_spreads=initial_spreads,
            random_state=random_state,
            prior=prior,
        )
    else:
        init_theta_log = np.asarray(init_theta_log, dtype=float)
        if init_theta_log.shape != (nwalkers, ndim):
            raise ValueError(f"init_theta_log must have shape {(nwalkers, ndim)}.")
        p0 = init_theta_log

    start_time = time.perf_counter()
    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_posterior_vectorized,
        args=(config, model, observations_list, prior),
        vectorize=True,
    )
    sampler.run_mcmc(p0, nsteps, progress=True)

    burn = int(burn_frac * nsteps)
    try:
        tau = sampler.get_autocorr_time(discard=burn, c=1)
        thin = int(np.min(tau) / 2) if np.min(tau) > 0 else 1
    except Exception:
        thin = 1

    flat_logp = sampler.get_log_prob(discard=burn, flat=True, thin=thin)
    flat_samples_log = sampler.get_chain(discard=burn, flat=True, thin=thin)
    flat_samples = 10**flat_samples_log
    flat_loglike = _batched_log_likelihood(
        flat_samples_log,
        model,
        observations_list,
        config,
        batch_size=postprocess_batch_size,
    )

    map_idx = int(np.argmax(flat_logp))
    mle_idx = int(np.argmax(flat_loglike))
    n_observations = observation_count(observations_list)
    n_parameters = ndim
    max_loglike = float(flat_loglike[mle_idx])
    bic = compute_bic(max_loglike, n_parameters, n_observations)
    acceptance_fraction = float(np.mean(sampler.acceptance_fraction))
    runtime_seconds = time.perf_counter() - start_time

    posterior_std_log = np.std(flat_samples_log, axis=0)

    results = {
        "samples": flat_samples,
        "samples_log": flat_samples_log,
        "logp": flat_logp,
        "loglike": flat_loglike,
        "parameter_names": list(config["names"]),
        "all_parameter_names": list(resolve_config_param_names(config)),
        "map": flat_samples[map_idx],
        "map_log": flat_samples_log[map_idx],
        "mle": flat_samples[mle_idx],
        "mle_log": flat_samples_log[mle_idx],
        "max_loglike": max_loglike,
        "bic": bic,
        "n_parameters": n_parameters,
        "n_observations": n_observations,
        "sampler": sampler,
        "thin": thin,
        "acceptance_fraction": acceptance_fraction,
        "fixed_params": dict(config.get("fixed_params", {})),
        "runtime_seconds": runtime_seconds,
        "posterior_std_log": posterior_std_log,
        "double_exp_prior": prior,
    }

    return results
