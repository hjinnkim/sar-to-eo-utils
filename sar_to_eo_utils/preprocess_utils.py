import numpy as np
import os

EPS = np.finfo(np.float32).eps

__IMAGE_TYPE__ = ['EO', 'SAR']
image_type_kwargs = {'image_type': 'EO'}


"""No data based"""
_image_type_kwargs = {'image_type': 'EO'}
_min_max_kwargs = {'minmax_min': 0, 'minmax_max': 2500}
_clip_kwargs = {'clip_min': 0, 'clip_max': 2500}
_clip_per_ch_kwargs = {'clip_B2_min': 0, 'clip_B2_max': 3720, 'clip_B3_min': 0, 'clip_B3_max': 3706, 'clip_B4_min': 0, 'clip_B4_max': 3552, }
_upper_kwargs = {'threshold': 5000}
_under_kwargs = {'threshold': 5000}
_norm_kwargs = {'norm_mean': np.array([0.09791994, 0.17582382, 0.10148141]), 'norm_std': np.array([0.00207213, 0.00210326, 0.00108511])}

def _normalize(arr: np.ndarray, image_type='EO', **kwargs) -> np.ndarray:
    _mean = _norm_kwargs['norm_mean']
    _std = _norm_kwargs['norm_std']
    try:
        _mean = kwargs['norm_mean']
    except:
        pass
    try:
        _std = kwargs['norm_std']
    except:
        pass
    return (arr-_mean)/_std
    

def _min_max(arr: np.ndarray, image_type='EO', **kwargs) -> np.ndarray:
    _arr = arr.astype(float)
    _min = _arr.min()
    _max = _arr.max()
    try:
        _min = kwargs['norm_min']
    except:
        pass
    try:
        _max = kwargs['norm_max']
    except:
        pass
    
    return ((_arr-_min)/(_max-_min+EPS)).round(8)

def _clip(arr: np.ndarray, image_type='EO', **kwargs) -> np.ndarray:
    if 'clip_per_ch' in kwargs and kwargs['clip_per_ch'] is True:
        assert image_type == 'EO'
        return _clip_per_ch(arr, **kwargs)
    
    _min = _clip_kwargs['clip_min']
    _max = _clip_kwargs['clip_max']
    try:
        _min = kwargs['clip_min']
    except:
        pass
    try:
        _max = kwargs['clip_max']
    except:
        pass
    
    return np.clip(arr, _min, _max)
    
    
def _clip_per_ch(arr:np.ndarray, **kwargs) -> np.ndarray:
    _B2_min = _clip_per_ch_kwargs['clip_B2_min']
    _B2_max = _clip_per_ch_kwargs['clip_B2_max']
    _B3_min = _clip_per_ch_kwargs['clip_B3_min']
    _B3_max = _clip_per_ch_kwargs['clip_B3_max']
    _B4_min = _clip_per_ch_kwargs['clip_B4_min']
    _B4_max = _clip_per_ch_kwargs['clip_B4_max']
    try:
        _B2_min = kwargs['clip_B2_min']
    except:
        pass
    try:
        _B2_max = kwargs['clip_B2_max']
    except:
        pass
    try:
        _B3_min = kwargs['clip_B3_min']
    except:
        pass
    try:
        _B3_max = kwargs['clip_B3_max']
    except:
        pass
    try:
        _B4_min = kwargs['clip_B4_min']
    except:
        pass
    try:
        _B4_max = kwargs['clip_B4_max']
    except:
        pass
    _arr = arr
    _arr[:, :, 0] = np.clip(_arr[:, :, 0], _B4_min, _B4_max)
    _arr[:, :, 1] = np.clip(_arr[:, :, 1], _B3_min, _B3_max)
    _arr[:, :, 2] = np.clip(_arr[:, :, 2], _B2_min, _B2_max)
    return _arr

def _upper(arr: np.ndarray, image_type='EO', **kwargs) -> np.ndarray:
    
    _threshold = _upper_kwargs['threshold']
    if 'threshold' in kwargs:
        _threshold = kwargs['threshold']
    
    if image_type == 'EO':
        return np.all(arr>_threshold, axis=2)
    
    if image_type == 'SAR':
        return arr>_threshold
    
def _under(arr: np.ndarray, image_type='EO', **kwargs) -> np.ndarray:
    
    _threshold = _under_kwargs['threshold']
    if 'threshold' in kwargs:
        _threshold = kwargs['threshold']
    
    if image_type == 'EO':
        return np.all(arr<_threshold, axis=2)
    
    if image_type == 'SAR':
        return arr<_threshold
    
def _no_data(arr: np.ndarray, image_type='EO', **kwargs) -> bool:
     
    _arr = arr.copy()
    if image_type == 'EO':
        return (np.all(_arr == [0, 0, 0], axis=2)).any()
    
    #TODO
    if image_type == 'SAR':
        return True    
    
    
__func_dict__ = {'min_max' : _min_max, 'clip': _clip, 'upper': _upper, 'under': _under, 'normalize': _normalize}
__filter_dict__ = {'no_data' : _no_data}

def preprocess(arr: np.ndarray, method: str or tuple(str) or list(str), **kwargs) -> np.ndarray:
        if type(method) == str:
            method = [method]
        for _method in method:
            assert _method in __func_dict__
            
        # check image type
        _image_type = image_type_kwargs['image_type']
        if 'image_type' in kwargs:
            _image_type = kwargs['image_type']
            del kwargs['image_type']
        assert _image_type in __IMAGE_TYPE__
        
        _arr = arr.copy()
        if _image_type == 'EO':
            assert len(_arr.shape) == 3 and _arr.shape[2] == 3
        
        if _image_type == 'SAR':
            assert len(_arr.shape) == 2
        
        for _method in method:
            _arr = __func_dict__[_method](_arr, image_type=_image_type, **kwargs)
            
        return _arr
    
def filtering(arr: np.ndarray, method: str or tuple(str) or list(str), **kwargs) -> bool:
    if type(method) == str:
        method = [method]
    for _method in method:
        assert _method in __filter_dict__
        
    # check image type
    _image_type = image_type_kwargs['image_type']
    if 'image_type' in kwargs:
        _image_type = kwargs['image_type']
        del kwargs['image_type']
    assert _image_type in __IMAGE_TYPE__
    
    _arr = arr.copy()
    if _image_type == 'EO':
        assert len(_arr.shape) == 3 and _arr.shape[2] == 3
    
    if _image_type == 'SAR':
        assert len(_arr.shape) == 2
    
    for _method in method:
        if __filter_dict__[_method](_arr, image_type=_image_type, **kwargs):
            return True
    return False
