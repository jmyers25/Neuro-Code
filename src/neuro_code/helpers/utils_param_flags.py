# src/neuro_code/helpers/utils_param_flags.py
import math

def is_fit_marker(v) -> bool:
    """
    Return True if a parameter value means 'fit this'.
    Treats both None and NaN as markers.
    """
    if v is None:
        return True
    if isinstance(v, float):
        try:
            return math.isnan(v)  # True only for NaN
        except Exception:
            return False
    return False
