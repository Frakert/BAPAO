"""Neural network surrogate models for physics simulations.

This module provides a SurrogateModel wrapper for parameterized neural networks
that predict physical quantities based on input parameters. It handles parameter
broadcasting, unit transformations, and model inference.

Key Classes:
  SurrogateModel: Wrapper for neural network-based surrogate models

Key Functions:
  _to_numpy: Convert TensorFlow/Keras outputs to NumPy arrays
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from BAPAO.parameters import PARAM_INDEX, PARAM_ORDER, canonicalize_param_names, model_columns_for


def _to_numpy(value) -> np.ndarray:
    """Convert TensorFlow/Keras tensor or array-like to NumPy array.

    Safely handles both tensor objects (with .numpy() method) and regular
    array-like objects, converting all to float32 NumPy arrays.

    Parameters
    ----------
    value : object
        A TensorFlow tensor, Keras output, or array-like object.

    Returns
    -------
    np.ndarray
        Converted NumPy array with dtype float32.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1, 2, 3])
    >>> _to_numpy(x)  # doctest: +SKIP
    array([1., 2., 3.], dtype=float32)
    """
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy(), dtype=np.float32)
    return np.asarray(value, dtype=np.float32)


class SurrogateModel:
    """Neural network surrogate model for physics-informed predictions.

    Wraps a trained neural network to evaluate physics models across parameter
    space. Handles parameter broadcasting, logarithmic transformations, and
    output post-processing.

    Attributes
    ----------
    model : object
        Trained neural network model (Keras/TensorFlow).
    active_param_names : Sequence[str]
        Names of parameters that the model varies over.
    model_columns : tuple[str, ...]
        Names of input columns/features expected by the neural network.

    Examples
    --------
    >>> surrogate = SurrogateModel(
    ...     model=trained_nn,
    ...     active_param_names=['d0', 's0'],
    ...     model_columns=['xi', 'd0', 's0']
    ... )
    >>> xi, theta = surrogate.evaluate(
    ...     xi=np.linspace(0, 1, 100),
    ...     d0=1.0,
    ...     s0=0.5
    ... )
    """

    def __init__(
        self,
        model,
        active_param_names: Sequence[str],
        model_columns: Sequence[str] | None = None,
    ):
        """Initialize SurrogateModel.

        Parameters
        ----------
        model : object
            Trained neural network (Keras/TensorFlow).
        active_param_names : Sequence[str]
            Names of parameters the model evaluates (e.g., ['d0', 's0']).
        model_columns : Sequence[str], optional
            Input feature names for the model. If None, automatically determined
            from active_param_names.

        Raises
        ------
        ValueError
            If model_columns is provided but incompatible with active_param_names.
        """
        self.model = model
        self.active_param_names = canonicalize_param_names(active_param_names)
        self.model_columns = tuple(model_columns or model_columns_for(self.active_param_names))

    def _broadcast_params(self, params: Mapping[str, object]) -> dict[str, np.ndarray]:
        """Broadcast parameters to common ensemble size.

        Converts all parameter values to float32 NumPy arrays and broadcasts
        scalars to match the largest parameter size (n_walkers). Ensures all
        parameters can be vectorized consistently.

        Parameters
        ----------
        params : Mapping[str, object]
            Dictionary of parameter names to values (scalars or array-like).

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary mapping parameter names to (n_walkers,) float32 arrays.

        Raises
        ------
        ValueError
            If no parameters provided, or if parameter sizes don't match
            (neither 1 nor n_walkers).
        """
        arrays = {
            name: np.atleast_1d(np.asarray(value, dtype=np.float32))
            for name, value in params.items()
        }
        if not arrays:
            raise ValueError("At least one parameter must be supplied.")

        n_walkers = max(array.size for array in arrays.values())
        broadcast = {}
        for name, array in arrays.items():
            if array.size == 1 and n_walkers > 1:
                array = np.repeat(array, n_walkers)
            elif array.size != n_walkers:
                raise ValueError(
                    f"Parameter '{name}' has length {array.size}, expected 1 or {n_walkers}."
                )
            broadcast[name] = array.astype(np.float32)

        return broadcast

    def evaluate(self, xi, **params):
        """Evaluate surrogate model across parameter ensemble.

        Evaluates the neural network at specified points (xi) for multiple
        parameter combinations. Handles parameter space transformations
        (e.g., log10 for rate constants), broadcasts ensemble sizes, and
        returns the model output in the same scale used during training.

        Parameters
        ----------
        xi : array-like
            Evaluation points (e.g., coverage values), shape (n_points,).
        **params : dict
            Parameter name-value pairs (e.g., d0=1.0, s0=0.5).
            Values can be scalars or arrays of length 1 or n_walkers.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            - xi : Unchanged evaluation points, shape (n_points,)
            - theta : Model predictions
              - Shape (n_points,) if single walker
              - Shape (n_walkers, n_points) if multiple walkers

        Raises
        ------
        ValueError
            If parameter broadcasting fails due to incompatible sizes.

        Examples
        --------
        >>> surrogate = SurrogateModel(model, ['d0', 's0'])
        >>> xi, theta = surrogate.evaluate(
        ...     xi=np.array([0.0, 0.5, 1.0]),
        ...     d0=np.array([1.0, 1.5]),  # 2 walkers
        ...     s0=0.5
        ... )
        >>> theta.shape  # doctest: +SKIP
        (2, 3)
        """
        xi = np.atleast_1d(np.asarray(xi, dtype=np.float32))
        broadcast_params = self._broadcast_params(params)

        n_walkers = len(next(iter(broadcast_params.values())))
        n_points = len(xi)

        xi_tiled = np.tile(xi, n_walkers)
        full_inputs = np.zeros((n_walkers * n_points, len(PARAM_ORDER)), dtype=np.float32)
        full_inputs[:, PARAM_INDEX["xi"]] = xi_tiled

        for name in PARAM_ORDER[1:]:
            if name not in broadcast_params:
                continue

            values = broadcast_params[name]
            if name == "k":
                values = np.log10(np.clip(values, 1e-30, None))

            full_inputs[:, PARAM_INDEX[name]] = np.repeat(values, n_points)

        selected_columns = [PARAM_INDEX["xi"], *[PARAM_INDEX[name] for name in self.model_columns]]
        inputs = full_inputs[:, selected_columns]

        out = self.model(inputs, training=False)
        out_np = _to_numpy(out)
        if out_np.ndim == 1:
            out_np = out_np.reshape(-1, 1)

        theta = out_np[:, 0].reshape(n_walkers, n_points)
        if n_walkers == 1:
            return xi, theta[0]
        return xi, theta
