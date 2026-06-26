"""Visualization functions for MCMC results and model comparisons.

This module provides plotting utilities for analyzing MCMC inference results,
including posterior distributions, chain traces, model comparisons, and
profile predictions. Uses Matplotlib with Agg backend for batch compatibility.

Key Functions:
  plot_pairwise_posteriors: Posterior joint and marginal distributions
  plot_chain_traces: MCMC sampler chain traces
  plot_profile_observations: Coverage profiles with observations/predictions
  plot_truth_vs_estimate: Parameter estimation accuracy scatter plots
  plot_bic_comparison: Bayesian Information Criterion model comparison
  plot_prediction_comparison: Predicted vs observed profiles
  plot_profile_fit: Single profile fit visualization
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.gridspec as gridspec

# Use a headless-safe backend so plotting works in tests and batch environments.
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from BAPAO.likelihoods import ProfileObservation, coerce_likelihood_term
from BAPAO.fitter import FitModule


def _finish_figure(fig, *, save_path=None, show_plot=True):
    """Complete figure rendering with optional save and display.

    Saves figure to disk if path provided, displays or closes based on
    show_plot flag, and returns the figure object.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to finalize.
    save_path : str or PathLike, optional
        File path to save figure to (300 DPI PNG). If None, no save.
    show_plot : bool, optional
        If True, display figure with plt.show(). If False, close figure.
        Default True.

    Returns
    -------
    matplotlib.figure.Figure
        The input figure object.
    """
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_pairwise_posteriors(
    results: Mapping[str, object] | None = None,
    *,
    samples: np.ndarray | None = None,
    parameter_names: Sequence[str] | None = None,
    truths: Mapping[str, float] | None = None,
    bins: int = 50,
    save_path=None,
    show_plot: bool = False,
):
    """Plot MCMC posterior distributions with adaptive visualization.

    For 1D: histogram. For 2D: KDE contours with marginals. For 3D: corner plot
    with HPD regions. For 4+D: corner plot with density contours.

    Can accept either:
    - Full results dict from run_mcmc() (backward compatible)
    - Individual samples array and parameter_names

    Parameters
    ----------
    results : Mapping[str, object], optional
        MCMC results dict with 'samples' and 'parameter_names'.
    samples : np.ndarray, optional
        Posterior samples, shape (n_samples, n_params). Used if results is None.
    parameter_names : Sequence[str], optional
        Parameter names. Used if results is None.
    truths : Mapping[str, float], optional
        Dictionary mapping parameter names to true values.
    bins : int, optional
        Number of histogram bins. Default 50.
    save_path : str or PathLike, optional
        File path to save figure.
    show_plot : bool, optional
        If True, display figure. Default True.

    Returns
    -------
    matplotlib.figure.Figure
        The generated posterior plot.
    """
    # Extract samples and names from either input format
    if results is not None:
        samples = np.asarray(results["samples"], dtype=float)
        parameter_names = list(
            results.get("parameter_names", [f"p{i}" for i in range(samples.shape[1])])
        )
    else:
        if samples is None or parameter_names is None:
            raise ValueError("Must provide either results dict OR (samples, parameter_names) tuple")
        samples = np.asarray(samples, dtype=float)
        parameter_names = list(parameter_names)

    n_params = samples.shape[1]

    map_array = results["map"]

    # Adaptive visualization based on parameter count
    if n_params == 1:
        # Single parameter: histogram
        fig, ax = plt.subplots(figsize=(6, 4))
        values = np.log10(np.maximum(samples[:, 0], 1e-12))
        ax.hist(values, bins=bins, color="steelblue", edgecolor="black")
        ax.set_xlabel(f"log10({parameter_names[0]})")
        ax.set_ylabel("Count")
        if truths and parameter_names[0] in truths:
            ax.axvline(np.log10(truths[parameter_names[0]]), color="red", linestyle="--")
        return _finish_figure(fig, save_path=save_path, show_plot=show_plot)

    elif n_params == 2:
        # Two parameters: fancy KDE with marginals
        return plot_posterior_2d(
            samples,
            parameter_names,
            truths=truths,
            map_array=map_array,
            bins=bins,
            save_path=save_path,
            show_plot=show_plot,
        )

    elif n_params >= 3:
        # Three parameters: corner plot with HPD regions
        return plot_posterior_3d_fancy(
            samples,
            parameter_names,
            truths=truths,
            bins=bins,
            save_path=save_path,
            show_plot=show_plot,
        )

    # old backup code for 4+ parameters: corner plot with density contours
    else:
        return plot_posterior_corner_density(
            samples,
            parameter_names,
            truths=truths,
            bins=bins,
            title=f"Joint Posterior: {n_params}D Inference",
            save_path=save_path,
            show_plot=show_plot,
        )

def plot_double_exponential_fit(
    xi_obs,
    theta_obs,
    popt,
    *,
    save_path=None,
    show_plot: bool = False,
):
    """Plot observed profile with fitted double exponential model.

    Parameters
    ----------
    xi_obs : array-like
        Normalized depth values.
    theta_obs : array-like
        Observed profile thickness values.
    popt : array-like
        Optimized parameters for the double exponential model.
    save_path : str or PathLike, optional
        Explicit save path override. If provided, overrides output_path.
    show_plot : bool, optional
        If True, display plot. Default True.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """

    fitter = FitModule()

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.scatter(xi_obs, theta_obs, label="Observed Data", s=20)
    ax.plot(
        xi_obs,
        fitter.double_exponential(xi_obs, *popt),
        label="Double Exp Fit",
        color="red",
    )

    ax.set_xlabel("Xi depth (-)")
    ax.set_ylabel("Profile thickness normalised (-)")
    ax.legend()
    ax.grid(True)

    return _finish_figure(fig, save_path=save_path, show_plot=show_plot)


def plot_profile_observations(
    observations: Sequence[object],
    *,
    truth_profiles: Sequence[Mapping[str, object]] | None = None,
    predicted_profiles: Sequence[Mapping[str, object]] | None = None,
    save_path=None,
    show_plot: bool = True,
):
    """Plot observed coverage profiles with optional truth and predictions.

    Creates subplots for each profile observation (extracted from likelihood
    terms). Overlays true physics simulations and model predictions if provided.

    Parameters
    ----------
    observations : Sequence[object]
        List of likelihood term objects (ProfileObservation or convertible).
    truth_profiles : Sequence[Mapping[str, object]], optional
        List of dicts with 'xi_obs' and 'theta_true' keys for truth curves.
    predicted_profiles : Sequence[Mapping[str, object]], optional
        List of dicts with 'xi_obs', 'theta_pred', 'label' for prediction curves.
    save_path : str or PathLike, optional
        File path to save figure.
    show_plot : bool, optional
        If True, display figure. Default True.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with profile subplots.

    Raises
    ------
    ValueError
        If no ProfileObservation terms found in observations.
    """
    profile_terms = []
    for observation in observations:
        term = coerce_likelihood_term(observation)
        if isinstance(term, ProfileObservation):
            profile_terms.append(term)
    if not profile_terms:
        raise ValueError("No profile observations were provided.")

    fig, axes = plt.subplots(
        1, len(profile_terms), figsize=(6 * len(profile_terms), 4), squeeze=False
    )

    for idx, term in enumerate(profile_terms):
        ax = axes[0, idx]
        ax.scatter(term.xi_obs, term.theta_obs, label="Observed", alpha=0.7)

        # if truth_profiles is not None and idx < len(truth_profiles):
        #     truth = truth_profiles[idx]
        #     ax.plot(truth["xi_obs"], truth["theta_true"], label="Physics truth", color="black")

        if predicted_profiles is not None and idx < len(predicted_profiles):
            pred = predicted_profiles[idx]
            ax.plot(
                pred["xi_obs"],
                pred["theta_pred"],
                label=pred.get("label", "Prediction"),
                color="tab:red",
            )

        ax.set_xlabel("normalised depth")
        ax.set_ylabel("normalised thickness")
        ax.set_title(term.name or f"profile_{idx + 1}")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.tight_layout()
    return _finish_figure(fig, save_path=save_path, show_plot=show_plot)


def plot_prediction_comparison(
    model,
    *,
    observations: Sequence[object],
    params: Mapping[str, float],
    save_path=None,
    show_plot: bool = False,
):
    """Plot model predictions against observations using learned parameters.

    Evaluates a physics/surrogate model at given parameters and plots
    predictions alongside observations for all profile observations.

    Parameters
    ----------
    model : object
        Physics or surrogate model with evaluate(xi, **params) method.
    observations : Sequence[object]
        List of likelihood terms (ProfileObservation or convertible).
    params : Mapping[str, float]
        Parameter values for model evaluation (e.g., {'d0': 1.0, 's0': 0.8}).
    save_path : str or PathLike, optional
        File path to save figure.
    show_plot : bool, optional
        If True, display figure. Default True.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with prediction comparison plots.

    Notes
    -----
    If observations have parameter transforms, they are applied before
    model evaluation.
    """
    predicted_profiles = []
    truth_profiles = []

    for observation in observations:
        term = coerce_likelihood_term(observation)
        if not isinstance(term, ProfileObservation):
            continue

        active_params = dict(params)
        if term.transform is not None:
            active_params.update(
                term.transform({name: np.asarray([value]) for name, value in params.items()})
            )
            active_params = {
                name: float(np.asarray(value).ravel()[0]) for name, value in active_params.items()
            }

        _, theta_pred = model.evaluate(
            term.xi_obs, **{name: np.asarray([value]) for name, value in active_params.items()}
        )
        predicted_profiles.append(
            {
                "xi_obs": term.xi_obs,
                "theta_pred": np.asarray(theta_pred, dtype=float).reshape(-1),
                "label": "Prediction",
            }
        )
        truth_profiles.append({"xi_obs": term.xi_obs, "theta_true": term.theta_obs})

    return plot_profile_observations(
        observations,
        truth_profiles=truth_profiles,
        predicted_profiles=predicted_profiles,
        save_path=save_path,
        show_plot=show_plot,
    )


def plot_truth_vs_estimate(
    rows: Sequence[Mapping[str, object]],
    *,
    parameter_names: Sequence[str],
    save_path=None,
    show_plot: bool = True,
):
    """Plot true vs estimated parameters from inference results.

    Creates scatter plots in log-log space comparing true parameter values
    (x-axis) against MAP estimates (y-axis). One subplot per parameter.
    Red dashed diagonal shows perfect agreement.

    Parameters
    ----------
    rows : Sequence[Mapping[str, object]]
        List of result dicts with keys '{name}_true' and '{name}_map' for each
        parameter name.
    parameter_names : Sequence[str]
        Parameter names to plot (must exist as keys in rows).
    save_path : str or PathLike, optional
        File path to save figure.
    show_plot : bool, optional
        If True, display figure. Default True.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with truth vs estimate scatter plots.

    Examples
    --------
    >>> rows = [
    ...     {'d0_true': 1.0, 'd0_map': 1.05, 's0_true': 0.8, 's0_map': 0.78},
    ...     {'d0_true': 2.0, 'd0_map': 1.95, 's0_true': 0.6, 's0_map': 0.62}
    ... ]
    >>> fig = plot_truth_vs_estimate(rows, parameter_names=['d0', 's0'])  # doctest: +SKIP
    """
    fig, axes = plt.subplots(
        1, len(parameter_names), figsize=(6 * len(parameter_names), 5), squeeze=False
    )

    for idx, name in enumerate(parameter_names):
        ax = axes[0, idx]
        x = np.asarray([row[f"{name}_true"] for row in rows], dtype=float)
        y = np.asarray([row[f"{name}_map"] for row in rows], dtype=float)
        ax.scatter(x, y, alpha=0.6)
        limits = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax.plot(limits, limits, "r--")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"True {name}")
        ax.set_ylabel(f"Predicted {name}")
        ax.set_title(f"{name}: truth vs estimate")
        ax.grid(True, which="both", alpha=0.2)

    fig.tight_layout()
    return _finish_figure(fig, save_path=save_path, show_plot=show_plot)


def plot_chain_traces(
    results: Mapping[str, object],
    *,
    save_path=None,
    show_plot: bool = False,
):
    """Plot MCMC chain traces (parameter evolution over iterations).

    Creates a subplot for each parameter showing all walker traces over
    MCMC iterations. Useful for diagnosing mixing and convergence.

    Parameters
    ----------
    results : Mapping[str, object]
        MCMC results dict containing 'sampler' object (from emcee) with
        get_chain() method, and optionally 'parameter_names'.
    save_path : str or PathLike, optional
        File path to save figure.
    show_plot : bool, optional
        If True, display figure. Default True.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with chain trace subplots.

    Raises
    ------
    ValueError
        If results dict does not contain a sampler object.

    Notes
    -----
    Sampler chains are expected in log10-space and are labeled accordingly.
    """
    sampler = results.get("sampler")
    if sampler is None:
        raise ValueError("Results do not include a sampler object for trace plotting.")

    chain = np.asarray(sampler.get_chain(), dtype=float)
    param_names = list(results.get("parameter_names", [f"p{i}" for i in range(chain.shape[-1])]))
    n_params = chain.shape[-1]

    fig, axes = plt.subplots(n_params, 1, figsize=(10, 3 * n_params), squeeze=False, sharex=True)
    steps = np.arange(chain.shape[0])

    for index, name in enumerate(param_names):
        ax = axes[index, 0]
        ax.plot(steps, chain[:, :, index], alpha=0.2, linewidth=0.7)
        ax.set_ylabel(f"log10({name})")
        ax.grid(True, alpha=0.2)

    axes[-1, 0].set_xlabel("Step")
    fig.tight_layout()
    return _finish_figure(fig, save_path=save_path, show_plot=show_plot)


def plot_bic_comparison(
    rows: Sequence[Mapping[str, object]],
    *,
    save_path=None,
    show_plot: bool = True,
):
    """Plot Bayesian Information Criterion (BIC) comparison across models.

    Creates a bar chart comparing BIC values for multiple models. Optional
    delta-BIC values displayed above bars for relative comparisons.

    Parameters
    ----------
    rows : Sequence[Mapping[str, object]]
        List of dicts with 'bic' key and optional 'label', 'name', 'delta_bic'.
    save_path : str or PathLike, optional
        File path to save figure.
    show_plot : bool, optional
        If True, display figure. Default True.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with BIC comparison bar chart.

    Raises
    ------
    ValueError
        If rows is empty.

    Notes
    -----
    Lower BIC is better. If 'label' or 'name' keys missing, defaults to
    'model_N' format.
    """
    if not rows:
        raise ValueError("At least one comparison row is required.")

    labels = [
        str(row.get("label", row.get("name", f"model_{index + 1}")))
        for index, row in enumerate(rows)
    ]
    bic_values = np.asarray([float(row["bic"]) for row in rows], dtype=float)
    delta_values = [row.get("delta_bic") for row in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        labels, bic_values, color=["steelblue", "indianred", "darkseagreen"][: len(labels)]
    )
    ax.set_ylabel("BIC")
    ax.set_title("Model Comparison by BIC")
    ax.grid(True, axis="y", alpha=0.2)

    for bar, bic_value, delta in zip(bars, bic_values, delta_values):
        label = f"{bic_value:.2f}"
        if delta is not None:
            label = f"{label}\nΔ={float(delta):.2f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    return _finish_figure(fig, save_path=save_path, show_plot=show_plot)


def plot_profile_fit(
    xi_obs,
    theta_obs,
    *,
    predictions: Sequence[Mapping[str, object]] | None = None,
    title: str | None = None,
    save_path=None,
    show_plot: bool = True,
):
    """Plot a single coverage profile with optional model predictions.

    Scatter plot of observed coverage (theta_obs) vs normalized depth (xi_obs),
    overlaid with optional prediction curves. Used for individual profile
    visualization and publication-quality figures.

    Parameters
    ----------
    xi_obs : array-like
        Normalized depth points, shape (N,), values in [0, 1].
    theta_obs : array-like
        Observed coverage values, shape (N,), values in [0, 1].
    predictions : Sequence[Mapping[str, object]], optional
        List of dicts with keys: 'xi_obs', 'theta_pred', 'label', optionally
        'linewidth' (default 2.0), 'linestyle' (default '-'), 'color'.
    title : str, optional
        Plot title. If None, no title shown.
    save_path : str or PathLike, optional
        File path to save figure.
    show_plot : bool, optional
        If True, display figure. Default True.

    Returns
    -------
    matplotlib.figure.Figure
        The profile fit figure.

    Examples
    --------
    >>> xi = np.linspace(0, 1, 100)
    >>> theta = 0.5 * xi
    >>> predictions = [{'xi_obs': xi, 'theta_pred': 0.48*xi, 'label': 'Model'}]
    >>> fig = plot_profile_fit(xi, theta, predictions=predictions)  # doctest: +SKIP
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(xi_obs, theta_obs, label="Observed", color="black", s=18, alpha=0.8)

    for prediction in predictions or []:
        ax.plot(
            prediction["xi_obs"],
            prediction["theta_pred"],
            label=str(prediction.get("label", "Prediction")),
            linewidth=float(prediction.get("linewidth", 2.0)),
            linestyle=str(prediction.get("linestyle", "-")),
            color=prediction.get("color"),
        )

    ax.set_xlabel("normalised depth")
    ax.set_ylabel("normalised thickness")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()
    return _finish_figure(fig, save_path=save_path, show_plot=show_plot)


def plot_posterior_2d(
    samples: np.ndarray,
    parameter_names: Sequence[str],
    *,
    truths: Mapping[str, float] | None = None,
    map_array: np.ndarray | None = None,
    bins: int = 120,
    save_path=None,
    show_plot: bool = True,
):
    """Plot 2D posterior with KDE contours and marginal distributions.

    Creates a 2x2 grid with joint KDE density plot and marginal histograms
    for a 2-parameter posterior. Uses GridSpec to align axes properly.

    Parameters
    ----------
    samples : np.ndarray
        Posterior samples, shape (n_samples, 2), in linear space.
    parameter_names : Sequence[str]
        Names of the two parameters.
    truths : Mapping[str, float], optional
        True parameter values for reference markers.
    bins : int, optional
        Number of histogram bins. Default 120.
    save_path : str or PathLike, optional
        File path to save figure.
    show_plot : bool, optional
        If True, display figure. Default True.

    Returns
    -------
    matplotlib.figure.Figure
        The generated 2D posterior plot.
    """
    samples = np.asarray(samples, dtype=float)
    if samples.shape[1] != 2:
        raise ValueError(f"samples must have shape (n_samples, 2), got {samples.shape}")

    # Convert to log10 space for plotting
    log_samples = np.log10(np.maximum(samples, 1e-12))
    log_x = log_samples[:, 0]
    log_y = log_samples[:, 1]

    # Find MAP (argmax of density approximation)
    if map_array is not None:
        map_log_x, map_log_y = (
            np.log10(np.maximum(map_array[0], 1e-12)),
            np.log10(np.maximum(map_array[1], 1e-12)),
        )
    else:
        try:
            kde = gaussian_kde(np.vstack([log_x, log_y]))
            sample_points = np.vstack([log_x, log_y])
            density_at_samples = kde(sample_points)
            map_idx = np.argmax(density_at_samples)
            map_log_x, map_log_y = log_x[map_idx], log_y[map_idx]
        except Exception:
            # Fallback if KDE fails
            map_log_x, map_log_y = np.median(log_x), np.median(log_y)

    # Construct grid for KDE visualization
    def get_padded_lims(data):
        vmin, vmax = np.min(data), np.max(data)
        rng = vmax - vmin
        return [vmin - 0.1 * rng, vmax + 0.1 * rng]

    xlim = get_padded_lims(log_x)
    ylim = get_padded_lims(log_y)

    xgrid = np.linspace(xlim[0], xlim[1], 100)
    ygrid = np.linspace(ylim[0], ylim[1], 100)
    X, Y = np.meshgrid(xgrid, ygrid)

    try:
        kde = gaussian_kde(np.vstack([log_x, log_y]))
        Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    except Exception:
        Z = np.ones_like(X)  # Fallback uniform contours

    # Create figure with GridSpec layout
    fig = plt.figure(figsize=(7, 7))
    gs = gridspec.GridSpec(
        2, 2, width_ratios=[3, 1], height_ratios=[1, 3], hspace=0.05, wspace=0.05
    )

    ax_joint = fig.add_subplot(gs[1, 0])
    ax_hist_x = fig.add_subplot(gs[0, 0], sharex=ax_joint)
    ax_hist_y = fig.add_subplot(gs[1, 1], sharey=ax_joint)

    # Joint density
    ax_joint.contourf(X, Y, Z, levels=20, cmap="Blues")
    ax_joint.set_xlabel(f"log₁₀({parameter_names[0]})")
    ax_joint.set_ylabel(f"log₁₀({parameter_names[1]})")
    ax_joint.set_xlim(xlim)
    ax_joint.set_ylim(ylim)

    # MAP marker
    ax_joint.plot(map_log_x, map_log_y, "o", color="orange", markersize=8, label="MAP")

    # True values if provided
    if truths is not None and len(parameter_names) >= 2:
        p0_true = truths.get(parameter_names[0])
        p1_true = truths.get(parameter_names[1])
        if p0_true is not None and p1_true is not None:
            ax_joint.plot(np.log10(p0_true), np.log10(p1_true), "r*", markersize=12, label="True")

    ax_joint.legend(loc="upper right")

    # Marginal histograms
    ax_hist_x.hist(log_x, bins=bins, color="gray", edgecolor="black", density=True)
    ax_hist_y.hist(
        log_y, bins=bins, orientation="horizontal", color="gray", edgecolor="black", density=True
    )

    # Hide redundant labels
    plt.setp(ax_hist_x.get_xticklabels(), visible=False)
    plt.setp(ax_hist_y.get_yticklabels(), visible=False)
    ax_hist_x.tick_params(left=False, labelleft=False)
    ax_hist_y.tick_params(bottom=False, labelbottom=False)

    fig.tight_layout()
    return _finish_figure(fig, save_path=save_path, show_plot=show_plot)


def plot_posterior_3d_fancy(
    samples: np.ndarray,
    parameter_names: Sequence[str],
    *,
    truths: Mapping[str, float] | None = None,
    bins: int = 60,
    save_path=None,
    show_plot: bool = True,
):
    """Plot 3D posterior with 2D density contours and HPD regions.

    Creates a 3x3 corner plot with 2D density contours and HPD (Highest Posterior
    Density) regions highlighted at 68% and 95% confidence levels. Diagonal shows
    marginal histograms.

    Parameters
    ----------
    samples : np.ndarray
        Posterior samples, shape (n_samples, 3), in linear space.
    parameter_names : Sequence[str]
        Names of the three parameters.
    truths : Mapping[str, float], optional
        True parameter values for reference markers.
    bins : int, optional
        Number of histogram bins. Default 60.
    save_path : str or PathLike, optional
        File path to save figure.
    show_plot : bool, optional
        If True, display figure. Default True.

    Returns
    -------
    matplotlib.figure.Figure
        The generated 3D posterior plot.
    """
    samples = np.asarray(samples, dtype=float)
    dimensions = samples.shape[1]

    # Work in log10 space
    log_samples = np.log10(np.maximum(samples, 1e-12))
    log_params = [log_samples[:, i] for i in range(dimensions)]
    param_labels = [f"log₁₀({name})" for name in parameter_names]

    fig, axes = plt.subplots(dimensions, dimensions, figsize=(14, 13))
    fig.suptitle("Joint Posterior: 3D Inference", fontsize=16, fontweight="bold")

    def plot_2d_density(ax, xlog, ylog):
        """Helper to create 2D density plot with HPD contours."""
        H, xedges, yedges = np.histogram2d(xlog, ylog, bins=bins)
        H = H.T

        if H.sum() == 0:
            return

        H_norm = H / H.sum()

        xcent = 0.5 * (xedges[:-1] + xedges[1:])
        ycent = 0.5 * (yedges[:-1] + yedges[1:])
        X, Y = np.meshgrid(xcent, ycent)

        # HPD computation
        Hflat = H_norm.flatten()
        idx = np.argsort(Hflat)[::-1]
        Hsorted = Hflat[idx]
        cumsum = np.cumsum(Hsorted)

        level_68 = Hsorted[np.searchsorted(cumsum, 0.68)] if len(Hsorted) > 0 else 0
        level_95 = Hsorted[np.searchsorted(cumsum, 0.95)] if len(Hsorted) > 0 else 0

        # Density shading
        ax.contourf(X, Y, H_norm, levels=40, cmap="Blues")

        # HPD contours
        ax.contour(X, Y, H_norm, levels=[level_95], colors="black", linewidths=2.0, linestyles="--")
        ax.contour(X, Y, H_norm, levels=[level_68], colors="navy", linewidths=3.0)

        ax.set_xlim(xlog.min(), xlog.max())
        ax.set_ylim(ylog.min(), ylog.max())

    # Diagonal: marginal histograms
    for i in range(dimensions):
        ax = axes[i, i]
        ax.hist(
            log_params[i],
            bins=bins,
            color="blue" if i == 0 else ("green" if i == 1 else "purple"),
            edgecolor="black",
        )
        ax.set_xlabel(param_labels[i])
        ax.set_ylabel("Count")

        # True lines
        if truths is not None and parameter_names[i] in truths:
            ax.axvline(
                np.log10(truths[parameter_names[i]]), color="red", linestyle="--", linewidth=2
            )

    # Off-diagonal: 2D density plots
    for i in range(dimensions):
        for j in range(dimensions):
            if i != j:
                ax = axes[i, j]
                plot_2d_density(ax, log_params[j], log_params[i])
                ax.set_xlabel(param_labels[j])
                ax.set_ylabel(param_labels[i])

                # True values
                if (
                    truths is not None
                    and parameter_names[j] in truths
                    and parameter_names[i] in truths
                ):
                    ax.plot(
                        np.log10(truths[parameter_names[j]]),
                        np.log10(truths[parameter_names[i]]),
                        "o",
                        color="red",
                        markersize=6,
                    )

    # Legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color="navy", lw=3, label="68% HPD"),
        Line2D([0], [0], color="black", lw=2, linestyle="--", label="95% HPD"),
    ]
    if truths is not None:
        legend_elements.append(
            Line2D(
                [0], [0], marker="o", color="w", markerfacecolor="red", markersize=6, label="True"
            )
        )

    fig.legend(handles=legend_elements, loc="upper right", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    return _finish_figure(fig, save_path=save_path, show_plot=show_plot)


def plot_posterior_corner_density(
    samples: np.ndarray,
    parameter_names: Sequence[str],
    *,
    truths: Mapping[str, float] | None = None,
    bins: int = 60,
    title: str | None = None,
    save_path=None,
    show_plot: bool = True,
):
    """Plot a corner posterior with density contours for all parameter pairs.

    Creates an N x N corner figure for N>=3 parameters. Diagonal cells show
    marginal histograms and off-diagonal cells show 2D density contours with
    HPD regions.

    Parameters
    ----------
    samples : np.ndarray
        Posterior samples, shape (n_samples, n_params), in linear space.
    parameter_names : Sequence[str]
        Names of the parameters.
    truths : Mapping[str, float], optional
        True parameter values for reference markers.
    bins : int, optional
        Number of histogram bins for marginals and density grids.
    title : str, optional
        Optional suptitle for the figure.
    save_path : str or PathLike, optional
        File path to save figure.
    show_plot : bool, optional
        If True, display figure. Default True.

    Returns
    -------
    matplotlib.figure.Figure
        The generated posterior corner plot.
    """
    samples = np.asarray(samples, dtype=float)
    n_params = samples.shape[1]
    if n_params < 3:
        raise ValueError(
            f"plot_posterior_corner_density requires at least 3 parameters, got {n_params}"
        )

    log_samples = np.log10(np.maximum(samples, 1e-12))
    param_labels = [f"log₁₀({name})" for name in parameter_names]

    fig, axes = plt.subplots(
        n_params, n_params, figsize=(4 * n_params, 4 * n_params), squeeze=False
    )
    if title is not None:
        fig.suptitle(title, fontsize=16, fontweight="bold")

    def plot_2d_density(ax, xlog, ylog):
        H, xedges, yedges = np.histogram2d(xlog, ylog, bins=bins)
        H = H.T

        if H.sum() == 0:
            return

        H_norm = H / H.sum()
        xcent = 0.5 * (xedges[:-1] + xedges[1:])
        ycent = 0.5 * (yedges[:-1] + yedges[1:])
        X, Y = np.meshgrid(xcent, ycent)

        Hflat = H_norm.flatten()
        idx = np.argsort(Hflat)[::-1]
        Hsorted = Hflat[idx]
        cumsum = np.cumsum(Hsorted)

        level_68 = Hsorted[np.searchsorted(cumsum, 0.68)] if len(Hsorted) > 0 else 0
        level_95 = Hsorted[np.searchsorted(cumsum, 0.95)] if len(Hsorted) > 0 else 0

        ax.contourf(X, Y, H_norm, levels=40, cmap="Blues")
        if level_95 > 0:
            ax.contour(
                X, Y, H_norm, levels=[level_95], colors="black", linewidths=1.5, linestyles="--"
            )
        if level_68 > 0:
            ax.contour(X, Y, H_norm, levels=[level_68], colors="navy", linewidths=2.0)

        ax.set_xlim(xlog.min(), xlog.max())
        ax.set_ylim(ylog.min(), ylog.max())

    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i, j]
            if i == j:
                ax.hist(log_samples[:, j], bins=bins, color="steelblue", edgecolor="black")
                if truths is not None and parameter_names[j] in truths:
                    ax.axvline(np.log10(truths[parameter_names[j]]), color="red", linestyle="--")
            else:
                plot_2d_density(ax, log_samples[:, j], log_samples[:, i])
                if (
                    truths is not None
                    and parameter_names[j] in truths
                    and parameter_names[i] in truths
                ):
                    ax.plot(
                        np.log10(truths[parameter_names[j]]),
                        np.log10(truths[parameter_names[i]]),
                        "o",
                        color="red",
                    )

            if i < n_params - 1:
                ax.set_xticklabels([])
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel(param_labels[j])

            if j > 0:
                ax.set_yticklabels([])
                ax.tick_params(labelleft=False)
            else:
                ax.set_ylabel(param_labels[i])

    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color="navy", lw=3, label="68% HPD"),
        Line2D([0], [0], color="black", lw=1.5, linestyle="--", label="95% HPD"),
    ]
    if truths is not None:
        legend_elements.append(
            Line2D(
                [0], [0], marker="o", color="w", markerfacecolor="red", markersize=6, label="True"
            )
        )

    fig.legend(handles=legend_elements, loc="upper right", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return _finish_figure(fig, save_path=save_path, show_plot=show_plot)
