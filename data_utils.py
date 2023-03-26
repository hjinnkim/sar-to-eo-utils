import json
import glob
from rasterio import open as r_open
import tqdm
from typing import Tuple

from preprocess_utils import *

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
        
def get_image_dict(path: str, prefix: str, filename_extension: str='tiff') -> list:
    assert prefix in __prefix__
    assert filename_extension in __postfix__
    
    _lst = glob.glob(os.path.join(path, f"{prefix}*.{filename_extension}"))
    
    _lst = [os.path.basename(_path) for _path in _lst]
    
    return {'path': path, 'channel': prefix.split('_')[1], 'filename_extension': filename_extension, 'list': _lst}

def save_json(path: str, name: str, data: list or dict):
    _path = f"{os.path.join(path, name)}.json"
    with open(_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"save {name} to the {_path}")
    
def load_json(path: str, name: str) -> list or dict:
    _path = f"{os.path.join(path, name)}"
    with open(_path, 'r') as f:
        _data = json.load(f)
    return _data

def sort_QA60_cloud_value(image_dict: dict) -> dict:
    
    assert image_dict['channel'] == 'QA60'
    
    _list = image_dict['list']
    _path = image_dict['path']   
    
    idx = 0
    
    from image_utils import save_image
    
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
    
    save_json('.', 'qa_value_dict_full', _dict)

    return {}

#TODO Sort image according to cloud ratio
def sort_QA60_cloud_ratio(image_dict: dict, ratio: float) -> dict:
    
    assert image_dict['channel'] == 'QA60'
    assert not ratio > 1.0 and ratio > 0.0
    
    _list = image_dict['list']
    _path = image_dict['path']
    _shape = image_open(_path, _list[0]).shape
    
    if ratio>1.0:
        print("ratio must be smaller than 1.0")
        return {}
    
    if ratio==0.0:
        return {'0.0': image_dict['list']}
    
    _ratio_edge = [0.0]
    while _ratio_edge[-1]<1.0:
        _ratio_edge.append(round(_ratio_edge[-1]+ratio, 3))
    
    
    _temp_list = [[]]*(len(_ratio_edge)-1)
    
    _num_px = _shape[0]*_shape[1]
    _ratio_edge = np.array(_ratio_edge)*_num_px

    return {}

def _get_loc_time(path: str) -> Tuple[str, str]:
    
    _temp = os.path.splitext(os.path.basename(path)) 
    
    return _temp[1], _temp[0].split('_')[2:]

def from_QA60_to_RGB(path: str, name: str, ) -> str:
    
    _postfix, _loc_time = _get_loc_time(name)
    _r_path = f"{'_'.join([__channel_info__['r']]+_loc_time)}{_postfix}"
    _g_path = f"{'_'.join([__channel_info__['g']]+_loc_time)}{_postfix}"
    _b_path = f"{'_'.join([__channel_info__['b']]+_loc_time)}{_postfix}"
    _r = image_open(path, _r_path)
    _g = image_open(path, _g_path)
    _b = image_open(path, _b_path)
    
    _rgb = np.dstack([_r, _g, _b])
        
    return _rgb