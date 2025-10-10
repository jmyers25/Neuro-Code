import copy
import numpy as np
from scipy.stats import truncnorm
from typing import Dict, List, Tuple

def sample_x0(pars: Dict[str, float]) -> List[float]:
    """
    Sample initial values (x0) for parameters marked as NaN in the input dictionary.
    Uses priors appropriate for each parameter.

    Args:
        pars (dict): Parameter dictionary, with np.nan indicating free parameters to fit.

    Returns:
        list: Initial sampled values for the free parameters.
    """
    pars_copy = copy.deepcopy(pars)
    x0 = []

    for key in sorted(pars_copy.keys()):
        if np.isnan(pars_copy[key]):
            if key in {"alpha", "alpha_neg", "alpha_pos"}:
                pars_copy[key] = np.random.beta(1.2, 1.2)
            elif key == "beta":
                pars_copy[key] = np.random.gamma(2, 1)
            elif key in {"exp", "exp_neg", "exp_pos"}:
                pars_copy[key] = np.random.beta(1.2, 1.2)
            elif key == "lossave":
                pars_copy[key] = truncnorm.rvs((0 - 2) / 2, (10 - 2) / 2, loc=2, scale=2)
            else:
                raise ValueError(f"Unknown parameter key: {key}")

            x0.append(pars_copy[key])

    return x0

import math

def _is_fit_value(v) -> bool:
    # Treat None OR NaN as "fit this"
    if v is None:
        return True
    if isinstance(v, (float, int)):
        try:
            return math.isnan(v)  # True only for NaN
        except Exception:
            return False
    return False

def get_bounds(pars: dict):
    """
    Return a list of (low, high) bounds aligned with the fitparams order from extract_pars().
    Defaults shown below; replace with your model-config defaults if you have them centralized.
    """
    # Example default bounds; adjust to your model conventions
    default_bounds = {
        "alpha":      (0.001, 1.0),
        "alpha_pos":  (0.001, 1.0),
        "alpha_neg":  (0.001, 1.0),
        "beta":       (1e-4,  10.0),
        "lossave":    (0.1,   2.0),
        "exp":        (0.5,   2.0),
        "exp_pos":    (0.5,   2.0),
        "exp_neg":    (0.5,   2.0),
    }

    parsed = extract_pars(pars)
    fitparams = parsed["fitparams"]

    bounds = []
    for key in fitparams:
        # choose configured bounds if present, else fallback default
        b = default_bounds.get(key)
        if b is None:
            # final fallback to something safe-ish
            b = (1e-6, 10.0)
        bounds.append(b)
    return bounds



#Need to fix Null handling -- When things are set to null, they are the parameters that should be fit/estimated
# WAnts to be looking at only the fitparams
# Make it just look at fitparams -- Just take fitparams
# Input will jsut be fitbounds to change the input to accept this