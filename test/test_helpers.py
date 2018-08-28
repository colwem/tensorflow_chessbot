
from helper_functions import split_by_fun
from helper_video import get_video_array
import numpy as np

def test_split_by_fun():

    # With python lists
    inputs, results = [], []

    inputs.append( ([1],) )
    results.append( [([1], 1)] )

    inputs.append( ([1, 2],) )
    results.append( [([1], 1), ([2], 2)] )

    inputs.append( ([1, 1, 2],) )
    results.append( [([1, 1], 1), ([2], 2)] )

    inputs.append( ([1, 1, 2, 2],) )
    results.append( [([1, 1], 1), ([2, 2], 2)] )

    inputs.append( ([1, 1 ,2, 2, 3],) )
    results.append( [([1, 1], 1), ([2, 2], 2), ([3], 3)] )


    for input, result in zip(inputs, results):
        assert split_by_fun(*input) == result

    # With numpy arrays
    inputs, results = [], []

    inputs.append( (np.asarray([1, 1 ,2, 2, 3]),) )
    results.append( [([1, 1], 1), ([2, 2], 2), ([3], 3)] )

    inputs.append( (np.asarray([1, 1, 1, 1, 1, 1, 1 ,2, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]),) )
    results.append( [([1, 1, 1, 1, 1, 1, 1], 1), ([2, 2], 2), ([3], 3), ([4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], 4)] )

    for input, result in zip(inputs, results):
        r = split_by_fun(*input)
        r = [(list(a), b) for a, b in r]
        assert r == result


def test_get_video_array():
    chunk_size = 1000
    arr, frames = next(get_video_array('test/testvideo.mp4', chunk_size), None)
    assert len(arr) == chunk_size
    assert len(frames) == chunk_size
    assert frames[500] == 500
    n = chunk_size//10
    for i in range(chunk_size // n):
        assert np.sum(arr[i * n]) > 1

