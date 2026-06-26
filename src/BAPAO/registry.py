"""Neural network surrogate model registry and factory.

Manages registration, discovery, and loading of pre-trained surrogate models.
Supports multiple model architectures and flexible parameter configurations
with automatic weight file discovery via configurable search paths.

Key Functions:
  build_surrogate: Main factory for creating SurrogateModel instances
  register_model_spec: Register a model specification with weights file
  register_model_instance: Register a pre-built model instance
  available_models: List all registered models
  default_model_directories: Get standard model search paths

Key Classes:
  ModelSpec: Specification for a surrogate model configuration
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from BAPAO.architectures import build_parameterized_dnn
from BAPAO.parameters import canonicalize_param_names, model_columns_for, resolve_config_param_names
from BAPAO.surrogates import SurrogateModel

ModelBuilder = Callable[[tuple[str, ...]], object]


@dataclass
class ModelSpec:
    """Specification for a registered surrogate model.

    Contains metadata and configuration for loading a neural network surrogate,
    including parameter names, input columns, and weight file location.

    Attributes
    ----------
    active_param_names : tuple[str, ...]
        Canonical parameter names the model handles (e.g., ('s0', 'k')).
    model_columns : tuple[str, ...]
        Input feature names for the neural network.
    builder : callable, optional
        Function to construct model from model_columns.
        Default: build_parameterized_dnn.
    weight_filename : str, optional
        Weight file name (search across model directories if set).
    description : str, optional
        Human-readable description of model.
    model_instance : object, optional
        Pre-built model instance (cached after first load).
    search_paths : list[Path], optional
        Custom directories to search for weight files.

    Examples
    --------
    >>> spec = ModelSpec(
    ...     active_param_names=('s0', 'k'),
    ...     model_columns=('xi', 's0', 'k'),
    ...     weight_filename='thermal_model.h5',
    ...     description='Thermal ALD surrogate'
    ... )
    """

    active_param_names: tuple[str, ...]
    model_columns: tuple[str, ...]
    builder: ModelBuilder = build_parameterized_dnn
    weight_filename: str | None = None
    description: str = ""
    model_instance: object | None = None
    search_paths: list[Path] = field(default_factory=list)

    def candidate_paths(self) -> list[Path]:
        """Get list of potential file paths to search for weights.

        Returns
        -------
        list[Path]
            Candidate paths: custom search_paths + default directories.
        """
        paths = list(self.search_paths)
        if self.weight_filename is None:
            return paths

        for model_dir in default_model_directories():
            paths.append(model_dir / self.weight_filename)

        return paths


def default_model_directories() -> list[Path]:
    """Get standard directories to search for surrogate model weights.

    Returns directories in order of precedence:
    1. Package models/ directory (src/BAPAO/models/)
    2. Repository models/ directory
    3. Sibling workspace directories (thermo-tool, plasma-tool, deeponet-surrogate)
    4. Custom directories from BAPAO_MODEL_DIRS environment variable

    Returns
    -------
    list[Path]
        List of unique directory paths to search for weight files.

    Notes
    -----
    Duplicates are automatically removed while preserving order.
    Set BAPAO_MODEL_DIRS environment variable to add custom directories
    (colon-separated on Unix, semicolon-separated on Windows).
    """
    package_root = Path(__file__).resolve().parent
    repo_root = package_root.parents[1]

    directories = [
        package_root / "models",
        repo_root / "models",
    ]

    extra_dirs = os.environ.get("BAPAO_MODEL_DIRS")
    if extra_dirs:
        directories.extend(Path(item) for item in extra_dirs.split(os.pathsep) if item)

    unique_directories: list[Path] = []
    seen = set()
    for directory in directories:
        if directory in seen:
            continue
        seen.add(directory)
        unique_directories.append(directory)
    return unique_directories


MODEL_REGISTRY: dict[tuple[str, ...], ModelSpec] = {}


def register_model_spec(
    param_names,
    *,
    model_columns: tuple[str, ...] | None = None,
    weight_filename: str | None = None,
    builder: ModelBuilder = build_parameterized_dnn,
    description: str = "",
    search_paths: list[Path] | None = None,
):
    """Register a surrogate model specification with weight file location.

    Adds a model to the registry for lazy loading. Weights are located
    automatically from standard directories or custom search_paths.

    Parameters
    ----------
    param_names : str or Sequence[str]
        Parameter names (e.g., 's0', 'k', 'd0'). Canonicalized and stored as tuple.
    model_columns : tuple[str, ...], optional
        Input feature names for neural network. If None, auto-generated from param_names.
    weight_filename : str, optional
        Filename of trained weights (e.g., 'model_thermal.h5'). If None,
        model_instance must be registered separately.
    builder : callable, optional
        Function to build model architecture. Default: build_parameterized_dnn.
    description : str, optional
        Human-readable description for listing/documentation.
    search_paths : list[Path], optional
        Custom directories to search for weight_filename.

    Examples
    --------
    >>> register_model_spec(
    ...     ('s0', 'k', 'd0'),
    ...     weight_filename='desorption_model.h5',
    ...     description='Desorption surrogate'
    ... )
    """
    key = canonicalize_param_names(param_names)
    MODEL_REGISTRY[key] = ModelSpec(
        active_param_names=key,
        model_columns=tuple(model_columns or model_columns_for(key)),
        builder=builder,
        weight_filename=weight_filename,
        description=description,
        search_paths=list(search_paths or []),
    )


def register_model_instance(
    param_names,
    model,
    *,
    model_columns: tuple[str, ...] | None = None,
    description: str = "",
):
    """Register a pre-built model instance directly.

    Useful for in-memory models or testing without weight files.
    The model instance is cached and reused across calls.

    Parameters
    ----------
    param_names : str or Sequence[str]
        Parameter names. Canonicalized to tuple key.
    model : object
        Pre-built neural network or model object with evaluate() method.
    model_columns : tuple[str, ...], optional
        Input feature names. If None, auto-generated from param_names.
    description : str, optional
        Human-readable description.

    Examples
    --------
    >>> nn = build_parameterized_dnn(('xi', 's0', 'k'))
    >>> register_model_instance(
    ...     ('s0', 'k'),
    ...     model=nn,
    ...     description='Custom thermal model'
    ... )
    """
    key = canonicalize_param_names(param_names)
    MODEL_REGISTRY[key] = ModelSpec(
        active_param_names=key,
        model_columns=tuple(model_columns or model_columns_for(key)),
        description=description,
        model_instance=model,
    )


def _load_model(spec: ModelSpec):
    """Load or retrieve cached neural network model from specification.

    If model_instance already exists in spec, returns it (caching).
    Otherwise, searches for weight file and loads it, then caches.

    Parameters
    ----------
    spec : ModelSpec
        Model specification with weight location and builder.

    Returns
    -------
    object
        Loaded neural network model with load_weights() method.

    Raises
    ------
    FileNotFoundError
        If no weight file specified and no model instance cached.
    FileNotFoundError
        If weight file not found in any candidate directory.
    """
    if spec.model_instance is not None:
        return spec.model_instance

    if spec.weight_filename is None:
        raise FileNotFoundError(
            f"No weight file is configured for parameter set {spec.active_param_names}. "
            "Register a trained model or add a matching weight file."
        )

    for candidate in spec.candidate_paths():
        if candidate.exists():
            model = spec.builder(spec.model_columns)

            # initialise with dummy
            dummy = np.zeros((1, len(spec.model_columns) + 1), dtype=np.float32)
            model(dummy)

            model.load_weights(candidate)
            spec.model_instance = model
            return model

    raise FileNotFoundError(
        f"No weights found for parameter set {spec.active_param_names}. "
        f"Looked for '{spec.weight_filename}' in {[str(path.parent) for path in spec.candidate_paths()]}"
    )


def build_surrogate(param_names, fixed_params: dict[str, float] | None = None) -> SurrogateModel:
    """Build a SurrogateModel from registry specifications.

    Main factory function for creating surrogate models. Looks up registered
    specs by parameter names, loads weights if needed, and returns wrapped
    SurrogateModel ready for inference.

    Parameters
    ----------
    param_names : str, Sequence[str], or dict
        Parameter names to look up. If dict (config), extracts param names.
        Canonicalized and combined with fixed_params for registry lookup.
    fixed_params : dict[str, float], optional
        Fixed parameters not varied during inference.

    Returns
    -------
    SurrogateModel
        Wrapped surrogate ready for evaluate(xi, **params) calls.

    Raises
    ------
    ValueError
        If parameter combination not registered. Lists available models.
    FileNotFoundError
        If registered model has no weights and not in-memory.

    Examples
    --------
    >>> config = {'names': ['s0', 'k'], 'bounds': {...}}
    >>> surrogate = build_surrogate(config)  # doctest: +SKIP
    >>> xi, theta = surrogate.evaluate(xi=np.array([0.5]), s0=0.8, k=50)  # doctest: +SKIP
    """
    if isinstance(param_names, dict):
        key = resolve_config_param_names(param_names)
    else:
        key = canonicalize_param_names([*param_names, *(fixed_params or {}).keys()])

    if key not in MODEL_REGISTRY:
        available = ", ".join(str(entry) for entry in sorted(MODEL_REGISTRY))
        raise ValueError(f"No surrogate model is registered for {key}. Available: {available}")

    spec = MODEL_REGISTRY[key]
    model = _load_model(spec)
    return SurrogateModel(
        model=model,
        active_param_names=key,
        model_columns=spec.model_columns,
    )


def available_models() -> dict[tuple[str, ...], str]:
    """List all registered surrogate models with descriptions.

    Returns
    -------
    dict[tuple[str, ...], str]
        Dictionary mapping parameter name tuples to descriptions.
        Empty description means "registered" (available).

    Examples
    --------
    >>> models = available_models()
    >>> for params, desc in models.items():
    ...     print(f"{params}: {desc}")  # doctest: +SKIP
    ('s0', 'k'): Bundled thermal surrogate
    ('s0', 'k', 'd0'): Bundled desorption surrogate
    """
    return {key: spec.description or "registered" for key, spec in MODEL_REGISTRY.items()}


register_model_spec(
    ("s0", "k"),
    weight_filename="model_final_lbfgs_dnn_thermal.weights.h5",
    description="Bundled thermal surrogate",
)
register_model_spec(
    ("s0", "k", "d0"),
    model_columns=("k", "s0", "d0"),
    weight_filename="model_final_lbfgs_dnn_desorp.weights.h5",
    description="Bundled desorption surrogate",
)
register_model_spec(
    ("s0", "k", "r"),
    weight_filename="model_final_lbfgs_dnn_plasma.weights.h5",
    description="Bundled plasma surrogate",
)
register_model_spec(
    ("s0", "k", "r", "d0"),
    model_columns=("k", "s0", "r", "d0"),
    weight_filename="model_final_lbfgs_dnn_desorp_recomb.weights.h5",
    description="Bundled plasma+desorption surrogate",
)
register_model_spec(
    ("s0", "k", "s_CVD"),
    model_columns=("k", "s0", "s_CVD"),
    weight_filename="model_final_lbfgs_dnn_thermal_cvd.weights.h5",
    description="Bundled thermal+CVD surrogate",
)
register_model_spec(
    ("s0", "k", "r", "s_CVD"),
    model_columns=("k", "s0", "r", "s_CVD"),
    weight_filename="model_final_lbfgs_dnn_plasma_cvd.weights.h5",
    description="Bundled plasma+CVD surrogate",
)
register_model_spec(
    ("s0", "k", "d0", "s_CVD"),
    model_columns=("k", "s0", "d0", "s_CVD"),
    weight_filename="model_final_lbfgs_dnn_desorp_cvd.weights.h5",
    description="Bundled desorption+CVD surrogate",
)
register_model_spec(
    ("s0", "k", "r", "d0", "s_CVD"),
    model_columns=("k", "s0", "r", "d0", "s_CVD"),
    weight_filename="model_final_lbfgs_dnn_desorp_recomb_cvd.weights.h5",
    description="Bundled plasma+desorption+CVD surrogate",
)
