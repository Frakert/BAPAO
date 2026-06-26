import csv
import math
import pickle
from pathlib import Path

import numpy as np


def uniform_log_sample(low, high, rng=None, size=None):
    """
    Generate samples from a uniform distribution in log-space.

    Parameters:
    low (float): The lower bound of the distribution (must be > 0).
    high (float): The upper bound of the distribution (must be > low).
    rng (np.random.Generator, optional): The random number generator to use.
    size (int or tuple of ints, optional): The number of samples to generate.
                                           If None, a single sample is returned.

    Returns:
    np.ndarray or float: Samples drawn from the uniform log-space distribution.
    """
    if low <= 0 or high <= low:
        raise ValueError("Low must be > 0 and high must be > low.")

    log_low = math.log(low)
    log_high = math.log(high)

    samples = (
        rng.uniform(log_low, log_high, size)
        if rng is not None
        else np.random.uniform(log_low, log_high, size)
    )

    return np.exp(samples)


def add_and_dump(filename, data_list):
    """
    Find a pickle file, open it, load the data, append new data to it, and dump it back.
    """
    try:
        with open(filename, "rb") as f:
            existing_data = pickle.load(f)
    except (FileNotFoundError, EOFError):
        existing_data = []
    existing_data.extend(data_list)
    with open(filename, "wb") as f:
        pickle.dump(existing_data, f)


def load_experimental_profile_csv(path) -> dict[str, object]:
    """Load thermal ALD profile from CSV file.

    Parses a CSV file with 'Distance' and 'Normalized signal' columns,
    extracting spatial points and coverage values for MCMC inference.

    Parameters
    ----------
    path : str or PathLike
        Path to CSV file. Expected columns: 'Distance', 'Normalized signal'.

    Returns
    -------
    dict[str, object]
        Dictionary with keys:
        - 'xi_obs': Spatial points (np.ndarray)
        - 'theta_obs': Normalized signal/coverage (np.ndarray)
        - 'dataset_name': Filename stem
        - 'source_path': Absolute path to CSV
        - 'num_points': Number of observations

    Raises
    ------
    ValueError
        If required columns missing or CSV empty or cannot be parsed.

    Examples
    --------
    >>> data = load_thermal_profile_csv('profile_data.csv')  # doctest: +SKIP
    >>> data['num_points']  # doctest: +SKIP
    42
    """
    csv_path = Path(path)
    xi_obs = []
    theta_obs = []

    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"No CSV header found in {csv_path}.")
        if "Distance" not in reader.fieldnames or "Normalized signal" not in reader.fieldnames:
            raise ValueError(
                f"{csv_path} must contain 'Distance' and 'Normalized signal' columns. "
                f"Found {reader.fieldnames}."
            )

        for row in reader:
            xi_obs.append(float(row["Distance"]))
            theta_obs.append(float(row["Normalized signal"]))

    xi = np.asarray(xi_obs, dtype=float)
    theta = np.asarray(theta_obs, dtype=float)

    if xi.size == 0:
        raise ValueError(f"{csv_path} did not contain any observations.")

    return {
        "dataset_name": csv_path.stem,
        "source_path": str(csv_path.resolve()),
        "xi_obs": xi,
        "theta_obs": theta,
        "num_points": int(xi.size),
    }
