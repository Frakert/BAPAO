from BAPAO._version import __version__
from BAPAO.constraints import (
    constraint_s0_gt_d0,
    constraint_s0_gt_s_cvd,
    default_physical_constraint,
)
from BAPAO.fitter import ALDType, FitModule, ModelType
from BAPAO.likelihoods import GaussianObservation, ProfileObservation
from BAPAO.mcmc import (
    compute_bic,
    log_likelihood_vectorized,
    log_posterior_vectorized,
    log_prior_vectorized,
    observation_count,
    resolve_parameter_batch,
    run_mcmc,
    unpack_theta,
)
from BAPAO.parameters import PARAM_ORDER
from BAPAO.physics import DoseBasedContinuumUnified, PhysicsModel
from BAPAO.plotting import (
    plot_bic_comparison,
    plot_chain_traces,
    plot_double_exponential_fit,
    plot_pairwise_posteriors,
    plot_prediction_comparison,
    plot_profile_fit,
    plot_profile_observations,
    plot_truth_vs_estimate,
)
from BAPAO.presets import (
    desorption_config,
    gaussian_observation,
    make_config,
    paired_profile_observations,
    plasma_config,
    single_profile_observation,
    thermal_config,
)
from BAPAO.priors import DoubleExpPrior, GaussianPrior, UniformPrior
from BAPAO.registry import (
    MODEL_REGISTRY,
    available_models,
    build_surrogate,
    register_model_instance,
    register_model_spec,
)
from BAPAO.surrogates import SurrogateModel
from BAPAO.utils import load_experimental_profile_csv

__all__ = [
    "__version__",
    # constraints
    "constraint_s0_gt_d0",
    "constraint_s0_gt_s_cvd",
    "default_physical_constraint",
    # fitter
    "ALDType",
    "FitModule",
    "ModelType",
    # likelihoods
    "GaussianObservation",
    "ProfileObservation",
    # mcmc
    "compute_bic",
    "log_likelihood_vectorized",
    "log_posterior_vectorized",
    "log_prior_vectorized",
    "observation_count",
    "resolve_parameter_batch",
    "run_mcmc",
    "unpack_theta",
    # parameters
    "PARAM_ORDER",
    # physics
    "DoseBasedContinuumUnified",
    "PhysicsModel",
    # plotting
    "plot_bic_comparison",
    "plot_chain_traces",
    "plot_double_exponential_fit",
    "plot_pairwise_posteriors",
    "plot_prediction_comparison",
    "plot_profile_fit",
    "plot_profile_observations",
    "plot_truth_vs_estimate",
    # presets
    "desorption_config",
    "gaussian_observation",
    "make_config",
    "paired_profile_observations",
    "plasma_config",
    "single_profile_observation",
    "thermal_config",
    # priors
    "DoubleExpPrior",
    "GaussianPrior",
    "UniformPrior",
    # registry
    "MODEL_REGISTRY",
    "available_models",
    "build_surrogate",
    "register_model_instance",
    "register_model_spec",
    # surrogates
    "SurrogateModel",
    # utils
    "load_experimental_profile_csv",
]
