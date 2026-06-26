"""Physical constraints for parameter inference.

Provides constraint functions that enforce physically-motivated inequalities
during MCMC sampling to reject unphysical parameter combinations.

Key Functions:
  constraint_s0_gt_d0: Enforce s0 > d0 (sticking > desorption)
  constraint_s0_gt_s_cvd: Enforce s0 > s_CVD (thermal > CVD sticking)
  default_physical_constraint: Enforce all applicable built-in constraints
"""

from __future__ import annotations

from collections.abc import Mapping


def constraint_s0_gt_d0(params: Mapping[str, float]) -> bool:
    """Enforce that thermal sticking probability exceeds desorption probability.

    Physical constraint for ALD processes: adsorbed species removal via
    desorption (d0) should be slower than adsorption (s0) for film growth.

    Parameters
    ----------
    params : Mapping[str, float]
        Parameter dictionary with keys 's0' and 'd0'.

    Returns
    -------
    bool
        True if constraint satisfied (s0 > d0), False otherwise.

    Examples
    --------
    >>> constraint_s0_gt_d0({'s0': 0.8, 'd0': 0.1})
    True
    >>> constraint_s0_gt_d0({'s0': 0.05, 'd0': 0.1})
    False
    """
    return float(params["s0"]) > float(params["d0"])


def constraint_s0_gt_s_cvd(params: Mapping[str, float]) -> bool:
    """Enforce that thermal sticking exceeds CVD/non-saturating sticking.

    Physical constraint for hybrid ALD-CVD processes: thermal adsorption
    (s0, saturating) should dominate over CVD adsorption (s_CVD, non-saturating).

    Parameters
    ----------
    params : Mapping[str, float]
        Parameter dictionary with keys 's0' and 's_CVD'.

    Returns
    -------
    bool
        True if constraint satisfied (s0 > s_CVD), False otherwise.

    Examples
    --------
    >>> constraint_s0_gt_s_cvd({'s0': 0.8, 's_CVD': 0.1})
    True
    >>> constraint_s0_gt_s_cvd({'s0': 0.05, 's_CVD': 0.1})
    False
    """
    return float(params["s0"]) > float(params["s_CVD"])


def default_physical_constraint(params: Mapping[str, float]) -> bool:
    """Apply all built-in constraints relevant to the supplied parameters.

    This helper is intentionally permissive for missing optional parameters:
    if a parameter is absent, its corresponding constraint is skipped. That
    makes it a safe default for mixed sweeps over different surrogate families.

    Parameters
    ----------
    params : Mapping[str, float]
        Parameter dictionary containing at least ``s0`` and any optional
        parameters that should be constrained.

    Returns
    -------
    bool
        True when all applicable constraints are satisfied.
    """
    if "d0" in params and not constraint_s0_gt_d0(params):
        return False
    if "s_CVD" in params and not constraint_s0_gt_s_cvd(params):
        return False
    return True
