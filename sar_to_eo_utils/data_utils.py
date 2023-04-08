import json
import glob
from rasterio import open as r_open
import tqdm
from typing import Tuple
import numpy as np
import os
from sklearn.model_selection import train_test_split

from sar_to_eo_utils.preprocess_utils import filtering, preprocess

__channel_info__ = {'r': 'Sentinel2_B4', 'g': 'Sentinel2_B3', 'b': 'Sentinel2_B2', 'cloud': 'Sentinel2_QA60'}

__prefix__ = ['Sentinel1_VV', 'Sentinel1_VH']+list(__channel_info__.values())

__postfix__ = ['tiff']

def image_open(path: str, name: str) -> np.ndarray:
    _path = os.path.join(path, name)
    try:
        with r_open(_path, 'r') as f:
            _arr = f.read(1)
        return _arr
    except Exception as e:
        print("Cannot open the path", e)
        
def get_image_dict_from_path(path: str, prefix: str, filename_extension: str='tiff') -> list:
    assert prefix in __prefix__
    assert filename_extension in __postfix__
    
    _lst = glob.glob(os.path.join(path, f"{prefix}*.{filename_extension}"))
    
    _lst = [os.path.basename(_path) for _path in _lst]
    
    return {'path': path, 'channel': prefix.split('_')[1], 'filename_extension': filename_extension, 'list': _lst}

def save_json(path: str, name: str, data: list or dict):
    _path = os.path.splitext(os.path.join(path, name))[0]+'.json'
    with open(_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"save {name} to the {_path}")
    
def load_json(path: str, name: str) -> list or dict:
    _path = f"{os.path.join(path, name)}"
    with open(_path, 'r') as f:
        _data = json.load(f)
    return _data

def sort_QA60_cloud_value(image_dict: dict, path: str, name: str, filter_method: str=None) -> None:
    
    assert image_dict['channel'] == 'QA60'
    
    
    _list = image_dict['list']
    _path = image_dict['path']   
        
    a = []
    b = []
    c = []
    d = []
    e = []
    f = []
    g = []
    
    _a = set([0])
    _b = set([1024])
    _c = set([2048])
    _d = set([0, 1024])
    _e = set([0, 2048])
    _f = set([1024, 2048])
    _g = set([0, 1024, 2048])
    
    for _qa_path in tqdm.tqdm(_list):
        _arr = image_open(_path, _qa_path)
        
        # filtering
        if filter_method is not None:
            _eo_arr = from_QA60_to_RGB(_path, _qa_path)
            if filtering(_eo_arr, method=filter_method, image_type='EO'):
                continue
            
        _arrlst = _arr.flatten().tolist()
        _tempset = set()
        for el in _arrlst:
            _tempset.add(el)
        
        if _tempset == _a:
            a.append(_qa_path)
        if _tempset == _b:
            b.append(_qa_path)
        if _tempset == _c:
            c.append(_qa_path)
        if _tempset == _d:
            d.append(_qa_path)
        if _tempset == _e:
            e.append(_qa_path)
        if _tempset == _f:
            f.append(_qa_path)
        if _tempset == _g:
            g.append(_qa_path)

    _dict = {}
    _dict['a'] = a
    _dict['b'] = b
    _dict['c'] = c
    _dict['d'] = d
    _dict['e'] = e
    _dict['f'] = f
    _dict['g'] = g
    
    save_json(path, name, _dict)

def _get_loc_time(path: str) -> Tuple[str, str]:
    
    _temp = os.path.splitext(os.path.basename(path)) 
    
    return _temp[1], _temp[0].split('_')[2:]

def eo_path_to_qa(path: str) -> str:
    _postfix, _loc_time = _get_loc_time(path)
    _qa_path = f"{'_'.join([__channel_info__['cloud']]+_loc_time)}{_postfix}"
    return _qa_path
    
    
def from_QA60_to_RGB(path: str, name: str) -> np.ndarray:
    
    _postfix, _loc_time = _get_loc_time(name)
    _r_path = f"{'_'.join([__channel_info__['r']]+_loc_time)}{_postfix}"
    _g_path = f"{'_'.join([__channel_info__['g']]+_loc_time)}{_postfix}"
    _b_path = f"{'_'.join([__channel_info__['b']]+_loc_time)}{_postfix}"
    _r = image_open(path, _r_path)
    _g = image_open(path, _g_path)
    _b = image_open(path, _b_path)
    
    _rgb = np.dstack([_r, _g, _b])
        
    return _rgb

def get_train_data_json(json_root: str, data_json: str, data_root: str, method: str or tuple(str) or list(str), clip_min: int=0, clip_max: int=2500):
    
    image_paths = load_json(json_root, data_json)
    
    _len = len(image_paths)
    
    channels_mean, channels_sqaured_mean = 0, 0
    
    for qa_path in tqdm.tqdm(image_paths):
        _eo = from_QA60_to_RGB(data_root, qa_path)
        _eo = preprocess(_eo, method=method, clip_min=clip_min, clip_max=clip_max, minmax_min=clip_min, minmax_max=clip_max)
        channels_mean += np.mean(_eo, axis=(0,1))
        channels_sqaured_mean += np.mean(np.square(_eo), axis=(0, 1))
    
    mean = (channels_mean / _len).tolist()
    std = np.sqrt(channels_sqaured_mean / _len - np.square(mean)).tolist()
    
    _dict = {}
    _dict['data_paths'] = image_paths
    _dict['data_root'] = data_root
    _dict['clip_min'] = clip_min
    _dict['clip_max'] = clip_max
    _dict['mean'] = mean
    _dict['std'] = std
    save_json(json_root, 'data_'+data_json, _dict)
    
def split_train_valid_data_json(json_root: str, data_json: str, seed: int=42):
    _dict = load_json(json_root, data_json)
    _paths = _dict['data_paths']
    
    _train_paths, _valid_paths = train_test_split(_paths, test_size=0.01, random_state=seed)
    
    _train_dict = _dict.copy()
    _train_dict['data_paths'] = _train_paths
    _valid_dict = _dict.copy()
    _valid_dict['data_paths'] = _valid_paths
    
    save_json(json_root, 'train_'+data_json, _train_dict)
    save_json(json_root, 'valid_'+data_json, _valid_dict)
    
    print(len(_train_dict['data_paths']))
    print(len(_valid_dict['data_paths']))
    
