"""
General utility functions
"""

import numpy as np

def is_positive(x):
    return (not is_array_like(x) and _is_numeric(x) and x > 0)

def is_positive_or_zero(x):
    return (not is_array_like(x) and _is_numeric(x) and x >= 0)

def is_array_like(x):
    return isinstance(x, (tuple, list, np.ndarray))

def is_positive_integer(x):
    return (not is_array_like(x) and _is_integer(x) and x > 0)

def is_string(x):
    return isinstance(x, str)

def _is_numeric(x):
    return isinstance(x, (float, int))

def _is_integer(x):
    return (_is_numeric(x) and (float(x) == int(x)))

def is_positive_integer_or_zero(x):
    return (not is_array_like(x) and _is_integer(x) and x >= 0)

def is_none(x):
    return isinstance(x, type(None))

def is_bool(x):
    return isinstance(x, bool)

def is_positive_array(x):
    return (is_array_like(x) and (x>0).all())

# custom exception to raise when we intentionally catch an error
class InputError(Exception):
    pass

