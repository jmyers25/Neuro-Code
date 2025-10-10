import copy
import json
import math
import numpy as np
import pandas as pd
from typing import Union

from neuro_code.helpers.utils_param_flags import is_fit_marker

def make_pars_dict(pars: Union[str, list]) -> dict:
    """
    Convert a JSON-like string or single-element list into a dictionary, 
    converting 'nan' strings to actual np.nan values.
    """
    if isinstance(pars, list) and pars:
        pars = pars[0]

    if isinstance(pars, str):
        try:
            pars = json.loads(pars.replace('nan', '"nan"').replace("'", '"'))
        except json.JSONDecodeError as e:
            raise ValueError(f"Could not parse parameters: {pars}") from e

    if not isinstance(pars, dict):
        raise TypeError("Input must be a JSON-formatted string or a dictionary.")

    for k, v in pars.items():
        if v == "nan":
            pars[k] = np.nan

    return pars

def extract_pars(pars: Union[str, dict]) -> dict:
    """
    Separate a parameter dictionary into fitparams (np.nan) and fixparams (concrete values).
    """
    if not isinstance(pars, dict):
        pars = make_pars_dict(pars)

    fitparams = []
    fixparams = []

    for key in sorted(pars.keys()):
        value = pars[key]
        if isinstance(value, float) and math.isnan(value):
            fitparams.append(key)
        else:
            fixparams.append(key)

    return {
        'fitparams': fitparams,
        'fixparams': fixparams
    }

def get_model_name(pars: Union[str, dict]) -> str:
    """
    Construct a model name string from a parameter dictionary, listing which params were fitted.
    """
    extracted = extract_pars(copy.deepcopy(pars))
    fitparams = extracted['fitparams']
    fixparams = extracted['fixparams']

    fit_str = '-'.join(fitparams) if fitparams else 'None'
    fix_str = '-'.join(fixparams) if fixparams else 'None'

    return f"LearningParams_Fit_{fit_str}_Fix_{fix_str}"
