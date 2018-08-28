import cv2
import numpy as np



def get_video_array(fn, chunk_size=1000):

    cap = cv2.VideoCapture(fn, 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    n = frame_count // chunk_size
    for i in range(n):
        current_frame_start = i * chunk_size
        length = min(frame_count - i * chunk_size, chunk_size)
        buf = np.empty((length, frame_height, frame_width), dtype=np.float32)

        for j in range(length):
            r = cap.read(0)[1]
            buf[j] = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
        yield buf, range(current_frame_start, current_frame_start + length)
