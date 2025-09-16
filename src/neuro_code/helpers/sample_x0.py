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

def get_bounds(pars: Dict[str, float]) -> Tuple[Tuple[float, float], ...]:
    """
    Return a tuple of (lower, upper) bounds for all parameters to be fitted (NaN in pars).

    Args:
        pars (dict): Parameter dictionary with np.nan for parameters to optimize.

    Returns:
        tuple: A tuple of bounds (min, max) for each free parameter, in the correct order.
    """
    bounds = []

    for key in sorted(pars.keys()):
        if np.isnan(pars[key]):
            if key in {"alpha", "alpha_neg", "alpha_pos", "exp", "exp_neg", "exp_pos"}:
                bounds.append((0.05, 2))
            elif key == "beta":
                bounds.append((0, 15))
            elif key == "lossave":
                bounds.append((0, 10))
            else:
                raise ValueError(f"Unknown parameter key: {key}")

    return tuple(bounds)


#Need to fix Null handling -- When things are set to null, they are the parameters that should be fit/estimated
# WAnts to be looking at only the fitparams
# Make it just look at fitparams -- Just take fitparams
# Input will jsut be fitbounds to change the input to accept this