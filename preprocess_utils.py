import numpy as np
import os

EPS = np.finfo(np.float32).eps

min_max_kwargs = {'min' : 0,
                  'max' : 1}

def _min_max(arr: np.ndarray, **kwargs) -> np.ndarray:
    _arr = arr.astype(float)
    _min = _arr.min()
    _max = _arr.max()
    try:
        _min = kwargs['min']
    except:
        pass
    try:
        _max = kwargs['max']
    except:
        pass
    
    return ((_arr-_min)/(_max-_min+EPS)).round(8)
    

__method__ = ['min_max',]
__func_dict__ = {'min_max' : _min_max}

def preprocess(arr: np.ndarray, method: str or tuple(str) or list(str), **kwargs) -> np.ndarray:
        if type(method) == str:
            method = [method]
        for _method in method:
            assert _method in __method__
        
        _arr = arr.copy()
        for _method in method:
            _arr = __func_dict__[_method](_arr, **kwargs)
            
        return _arr