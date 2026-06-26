"""
Fitting of the double exponential and sigmoid functions to ALD thickness profiles,
with extraction of physical parameters.

Example usage:
```Python
fitter = FitModule()
popt, results = fit_profile(
        xi_obs,
        theta_obs,
        model_type = ModelType.DOUBLE_EXP_STABLE,
        ald_type = ALDType.THERMAL)

"""
from enum import Enum

import numpy as np
from scipy.optimize import curve_fit


class ModelType(Enum):
    DOUBLE_EXP = ("Physical (Double Exp)", "g-")
    DOUBLE_EXP_STABLE = ("Physical (Stable x50)", "b-")  # New Stable Option
    SIGMOID = ("Empirical (Sigmoid)", "r--")

    def __init__(self, label, style):
        self.label = label
        self.style = style

    @property
    def func(self):
        mapping = {
            ModelType.DOUBLE_EXP: FitModule.double_exponential,
            ModelType.DOUBLE_EXP_STABLE: FitModule.double_exponential_stable,
            ModelType.SIGMOID: FitModule.boltzmann_sigmoid,
        }
        return mapping[self]

    @property
    def initial_guess(self):
        mapping = {
            ModelType.DOUBLE_EXP: [1.0, 50.0],
            ModelType.DOUBLE_EXP_STABLE: [0.2, 50.0],  # Initial x50 at middle, B=50
            ModelType.SIGMOID: [1, 0, 0.5, 0.1],
        }
        return mapping[self]


class ALDType(Enum):
    THERMAL = 1
    PLASMA = 2


class FitModule:
    AR = 1000

    @staticmethod
    def double_exponential(x:float|np.ndarray, A:float, B:float) -> float|np.ndarray:
        """Original formulation: sensitive to A exploding."""
        return 1 - np.exp(-A * np.exp(-B * x))

    @staticmethod
    def double_exponential_stable(x:float|np.ndarray, x50:float, B:float):
        """
        Stable formulation fitting the front position (x50) directly.
        Derived from: A = ln(2) * exp(B * x50)
        """
        # We use a small epsilon to prevent log(0) or exp(large) issues
        return 1 - np.exp(-np.log(2) * np.exp(-B * (x - x50)))

    @staticmethod
    def boltzmann_sigmoid(x, d_top, d_base, x50, w):
        return d_base + (d_top - d_base) / (1 + np.exp((x - x50) / w))

    def _determine_values_thermal(self, A, B):
        """Standard extraction for Thermal ALD."""
        sticking_prob = (4 / 3) * (1 / 0.79) * (B / self.AR) ** 2
        # Apply the 0.79 correction to k as well for consistency with s0
        dim_dose = (A * 0.79) / (B**2)
        return sticking_prob, dim_dose

    def _determine_values_plasma(self, A, B):
        """Standard extraction for Plasma ALD."""
        stick_recomb = (4 / 3) * (B / self.AR) ** 2
        stick_dose = (4 * A) / (3 * self.AR**2)
        return stick_recomb, stick_dose

    def _determine_value_from_sigmoid_fit(self, popt):
        d_top, d_base, x50, w = popt
        slope_at_50 = abs((d_top - d_base) / (4 * w))
        slope = slope_at_50 / self.AR
        return slope**2 * 13.9  # K. Arts approximation

    def fit_profile(
        self,
        depth,
        thickness,
        model_type: ModelType = ModelType.DOUBLE_EXP_STABLE,
        ald_type: ALDType = ALDType.THERMAL,
    ):
        try:
            # Set bounds to prevent physical impossibility
            # x50 can be slightly outside [0,1] if the front hasn't entered or has passed
            lower_bounds = [0] * len(model_type.initial_guess)
            if model_type == ModelType.DOUBLE_EXP_STABLE:
                lower_bounds = [
                    0.0,
                    0.1,
                ]  # x50 can be negative if front is at entrance

            popt, _ = curve_fit(
                model_type.func,
                depth,
                thickness,
                p0=model_type.initial_guess,
                bounds=(lower_bounds, np.inf),
            )
        except RuntimeError:
            return None, {}

        physical_results = {}

        # Determine A and B for physical extraction
        if model_type == ModelType.DOUBLE_EXP_STABLE:
            x50, B = popt[0], popt[1]
            print(f"Fitted x50: {x50}, B: {B}")
            # Convert back to A: A = ln(2) * exp(B * x50)
            A = np.log(2) * np.exp(B * x50)
            popt = A, B  # bring back in expected format
        elif model_type == ModelType.DOUBLE_EXP:
            A, B = popt[0], popt[1]
        else:
            A, B = None, None

        if A is not None:
            if ald_type == ALDType.THERMAL:
                s, dose = self._determine_values_thermal(A, B)
                physical_results = {"sticking_prob": s, "dim_dose": dose}
            else:
                sr, sd = self._determine_values_plasma(A, B)
                physical_results = {"stick_recomb": sr, "stick_dose": sd}
        elif model_type == ModelType.SIGMOID:
            s0_r = self._determine_value_from_sigmoid_fit(popt)
            physical_results = {"est_s0_r": s0_r}

        return popt, physical_results


# Example usage:
# fitter = FitModule()
# fitter.compare_models(xi_data, thickness_data, ALDType.THERMAL)
