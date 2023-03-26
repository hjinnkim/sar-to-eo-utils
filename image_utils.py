import matplotlib.pyplot as plt

from preprocess_utils import *

__GRAY_CHANNEL__ = ['VV', 'VH', 'QA60']
__EO_DICT__={'B2':"Blues", 'B3':"Greens", 'B4':"Reds"}
__RGB_CHANNEL__ = ['RGB']
__FORMAT__ = ['jpg', 'png']


#TODO apply filter to the arr
#TODO or check if arr is filtered
def visualize_image(arr: np.ndarray, channel: str, name='') -> None:
    if channel in __GRAY_CHANNEL__:
        _cmap = 'gray'
    elif channel in __EO_DICT__.keys():
        _cmap = __EO_DICT__[channel]
    elif channel in __RGB_CHANNEL__:
        _cmap = 'jet'
    else:
        print("Invalid channel input, failed to visualize")
        return
    
    plt.figure(figsize=(6, 6))
    plt.title(name)
    plt.imshow(arr, cmap=_cmap, vmin=0.0, vmax=1.0)
    plt.show()

#TODO apply filter to the arr
#TODO or check if arr is filtered
def save_image(arr: np.ndarray, savepath: str, channel: str, name: str, format: str='jpg') -> None:
    
    if channel in __GRAY_CHANNEL__:
        _cmap = 'gray'
    elif channel in __EO_DICT__.keys():
        _cmap = __EO_DICT__[channel]
    elif channel in __RGB_CHANNEL__:
        assert len(arr.shape) == 3 and arr.shape[2] == 3
        _cmap = 'jet'
    else:
        print("Invalid channel input, failed to save")
        return
    
    if format in __FORMAT__:
        _format = format
    else:
        print("Invalid format, failed to save")
        return
    
    _name = os.path.splitext(name)[0]+f"_{channel}"
    
    if not os.path.exists(savepath):
        os.makedirs(savepath)
      
    plt.imsave(f"{os.path.join(savepath, _name)}.{_format}", arr, cmap=_cmap, format=_format, vmin=0.0, vmax=1.0)
    print(f"\tsave {_name} to {os.path.join(savepath, _name)}.{_format}")