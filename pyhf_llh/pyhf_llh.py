import os 
import json 
import pyhf 

import numpy as np 
import jax.numpy as jnp
from jax import grad, jit 

from math import inf 
from typing import Callable, Tuple


# Makro to determine if jax is activated for pyhf
is_jax_backend = lambda: type(pyhf.get_backend()[0]) == pyhf.tensor.jax_backend


def load_pyhf(ws:str) -> Tuple[pyhf.Model, np.ndarray, np.ndarray]:
    """Load a pyhf workspace and return model, main_data and data."""
    assert os.path.exists(ws) 

    print(f"load workspace '{ws}'")

    with open(ws) as serialized:
        spec = json.load(serialized)

    workspace = pyhf.Workspace(spec)
    model = workspace.model()

    main_data = workspace.data(model, include_auxdata=False)
    main_data = np.array(main_data)
    
    data = np.array(list(main_data) + model.config.auxdata)
    
    return model, main_data, data


def pyhf_llh(ws: str) -> Callable: 
    """
    Returns the log likelihood function from the pyhf workspace 'ws'.
    
    Only use this function if you want a Python likelihood without jax.jit 
    >>> llh = pyhf_llh('path/to/ws.json')
    
    For a 'jax' likelihood use 
    >>> llh, llh_with_grad = pyhf_llh_with_grad('path/to/ws.json')
    """
    model, main_data, _ = load_pyhf(ws)
    logpdf = model.main_model.logpdf

    def llh(param: np.ndarray) -> jnp.DeviceArray:  # or -> float 
        """pyhf log likelihood from the main_model."""
        return logpdf(main_data, param)
    
    return jit(llh) if is_jax_backend() else llh 


def pyhf_llh_with_grad(ws:str) -> Tuple[Callable, Callable]:
    """
    Returns the Tuple (llh, llh_with_grad) for the pyhf workspace 'ws'
    that is required for PyCallDensityWithGrad. 
    
    For pyhf the backend must be set to 'jax' for AD. 
    """
    
    assert is_jax_backend(), "'pyhf.set_backend('jax')' required"

    # jitted llh and compute grad
    llh = pyhf_llh(ws)
    llh_grad = jit(grad(llh))

    # convert to float and catch out 'nan' values of pyhf
    llh_float = lambda x: -inf if jnp.isnan(llh(x).block_until_ready()) else float(llh(x)) 

    def llh_with_grad(param: np.ndarray) -> Tuple[float, np.ndarray]:
        """Return (llh, llh_grad) and convert jax.DeviceArray to (float, np.float64)"""
        return (llh_float(param), np.array(llh_grad(param).block_until_ready(), dtype=np.float64))

    return (llh_float, llh_with_grad)