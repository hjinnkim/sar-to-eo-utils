from data_utils import *
from image_utils import *
from preprocess_utils import *
import sys
np.set_printoptions(threshold=sys.maxsize)



if __name__=="__main__":
    base_dir = "/home/khj/SAR-to-EO"
    sent1_vv_dir = "sent1_vv"
    sent1_vh_dir = "sent1_vh"
    sent2_dir = "sent2_qa_b1-b4"
    save_dir = "result"
    
    vv_prefix = "Sentinel1_VV"
    vh_prefix = "Sentinel1_VH"
    b2_prefix = "Sentinel2_B2"
    b3_prefix = "Sentinel2_B3"
    b4_prefix = "Sentinel2_B4"
    qa_prefix = "Sentinel2_QA60"
    
    vv = "VV"
    vh = "VH"
    qa = "QA60"
    b2 = "B2"
    b3 = "B3"
    b4 = "B4"
    
    
    """
    QA60 image들을 읽고 list 형태로 저장
    """
    # QA60_dict = get_image_dict(sent2_dir, qa_prefix)

    """
    QA60 image들을 포함된 QA60 band value 조합에 따라 분류 -> qa_value_dict_full.json 저장
    
    _a = set([0])
    _b = set([1024])
    _c = set([2048])
    _d = set([0, 1024])
    _e = set([0, 2048])
    _f = set([1024, 2048])
    _g = set([0, 1024, 2048])
    
    """
    # sort_QA60_cloud_value(QA60_dict, 0.1)
    
    
    """
    QA60 image에 대응되는 EO image에 Total min-max / Per Channel min-max를 적용한 결과를 저장
    
    """
    ## Total min-max
    dir = 'qa_value_dict_min_max'
    _dict = load_json('.', 'qa_value_dict_full.json')
    for key in _dict.keys():
        save_dir = os.path.join(dir, key)
        
        
        length = min(3000, len(_dict[key]))
        
        for path in _dict[key][:length]:
            _arr = image_open(sent2_dir, path)
            _arr = preprocess(_arr, 'min_max', min=0, max=2048)
            save_image(_arr, save_dir, 'QA60', path)
            _arr = from_QA60_to_RGB(sent2_dir, path)
            _arr = preprocess(_arr, 'min_max')
            save_image(_arr, save_dir, 'RGB', path)
    
    ## Per-Channel min-max
    # dir = 'qa_value_dict_min_max_per_ch' 
    # _dict = load_json('.', 'qa_value_dict_full.json')
    # for key in _dict.keys():
    #     save_dir = os.path.join(dir, key)
        
        
    #     length = min(3000, len(_dict[key]))
        
    #     for path in _dict[key][:length]:
    #         _arr = image_open(sent2_dir, path)
    #         _arr = preprocess(_arr, 'min_max', min=0, max=2048)
    #         save_image(_arr, save_dir, 'QA60', path)
    #         _arr = from_QA60_to_RGB(sent2_dir, path)
    #         _arr[:, :, 0] = preprocess(_arr[:, :, 0], 'min_max')
    #         _arr[:, :, 1] = preprocess(_arr[:, :, 1], 'min_max')
    #         _arr[:, :, 2] = preprocess(_arr[:, :, 2], 'min_max')
    #         save_image(_arr, save_dir, 'RGB', path)