from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

PARAM_ORDER = ["xi", "k", "s0", "r", "d0", "s_CVD"]
USER_PARAM_ORDER = ["s0", "k", "r", "d0", "s_CVD"]
REQUIRED_PARAMS = ("s0", "k")
PARAM_INDEX = {name: idx for idx, name in enumerate(PARAM_ORDER)}
USER_PARAM_INDEX = {name: idx for idx, name in enumerate(USER_PARAM_ORDER)}


def canonicalize_param_names(names: Sequence[str]) -> tuple[str, ...]:
    unique_names = list(dict.fromkeys(names))
    unknown = sorted(set(unique_names) - set(USER_PARAM_ORDER))
    if unknown:
        raise ValueError(f"Unknown parameters: {unknown}")

    missing = [name for name in REQUIRED_PARAMS if name not in unique_names]
    if missing:
        raise ValueError(f"Missing required parameters: {missing}")

    return tuple(name for name in USER_PARAM_ORDER if name in unique_names)


def model_columns_for(names: Sequence[str]) -> tuple[str, ...]:
    canonical_names = canonicalize_param_names(names)
    return tuple(name for name in PARAM_ORDER[1:] if name in canonical_names)


def resolve_config_param_names(config: Mapping[str, object]) -> tuple[str, ...]:
    names = list(config.get("names", ()))  # type: ignore[arg-type]
    fixed_params = config.get("fixed_params", {})
    if not isinstance(fixed_params, Mapping):
        raise TypeError("Config 'fixed_params' must be a mapping when provided.")
    return canonicalize_param_names([*names, *fixed_params.keys()])


def _validate_positive_parameter(name: str, value: Any):
    if value is None:
        raise ValueError(f"Fixed parameter '{name}' cannot be None.")

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for entry in value:
            _validate_positive_parameter(name, entry)
        return

    if float(value) <= 0:
        raise ValueError(f"Fixed parameter '{name}' must be strictly positive.")


def validate_config(config: Mapping[str, object]) -> dict[str, object]:
    if "names" not in config:
        raise ValueError("Config must contain a 'names' entry.")
    if "bounds" not in config:
        raise ValueError("Config must contain a 'bounds' entry.")

    param_names = list(config["names"])  # type: ignore[arg-type]
    fixed_params = config.get("fixed_params", {})
    if not isinstance(fixed_params, Mapping):
        raise TypeError("Config 'fixed_params' must be a mapping when provided.")
    overlap = sorted(set(param_names) & set(fixed_params))
    if overlap:
        raise ValueError(f"Parameters cannot be both sampled and fixed: {overlap}")

    resolve_config_param_names({"names": param_names, "fixed_params": fixed_params})

    bounds = config["bounds"]
    if not isinstance(bounds, Mapping):
        raise TypeError("Config 'bounds' must be a mapping of parameter names to tuples.")

    for name in param_names:
        if name not in bounds:
            raise ValueError(f"Missing bounds for parameter '{name}'.")
        lo, hi = bounds[name]  # type: ignore[index]
        if lo <= 0 or hi <= 0:
            raise ValueError(f"Bounds for '{name}' must be strictly positive.")
        if lo >= hi:
            raise ValueError(f"Bounds for '{name}' must satisfy lo < hi.")

    for name, value in fixed_params.items():
        if name not in USER_PARAM_ORDER:
            raise ValueError(f"Unknown fixed parameter '{name}'.")
        _validate_positive_parameter(name, value)

    normalized = dict(config)
    normalized["names"] = param_names
    normalized["bounds"] = dict(bounds)
    normalized["fixed_params"] = dict(fixed_params)
    return normalized
