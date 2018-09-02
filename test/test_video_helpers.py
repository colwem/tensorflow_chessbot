from helper_functions import split_by_fun
from video_helpers import tile
import numpy as np

def test_tile():

    a = np.arange(3 * 2 * 2).reshape((3, 2, 2))
    print(a)
    tiled = tile(((a, a), (a, a)), title_bar_height=10, padding=1, add_frame_num=False)
    print(tiled)

