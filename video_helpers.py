import cv2
import numpy as np
import cairosvg
from chess_helpers import board_arrays_to_svgs


class VideoContainer(object):

    def __init__(self, fn):
        self.fn = fn
        self._current_frame = 0
        self._initialize_capture(self.fn)

    def get_video_array(self, chunk_size=1000):

        n = self.frame_count // chunk_size

        for i in range(n):
            current_frame_start = i * chunk_size
            length = min(self.frame_count - i * chunk_size, chunk_size)
            buf = np.empty((length, self.frame_height, self.frame_width), dtype=np.float32)

            for j in range(length):
                r = self._cap.read(0)[1]
                buf[j] = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
                self._current_frame += 1
            yield buf, current_frame_start

    def get_frame_at(self, frame_number):
        current_frame = self._current_frame
        self.seet_to(frame_number)
        img = self.read()
        self.seek_to(current_frame)
        return img

    def _reinitialize(self):
        self._cap.release()
        self._initialize_capture(self.fn)

    def _initialize_capture(self, fn):
        self._cap = cv2.VideoCapture(fn, 0)
        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_rate = self._cap.get(cv2.CAP_PROP_FPS)

    def seek_to(self, frame):
        if self._current_frame < frame:
            while self._current_frame < frame - 1:
                self.read()
        else:
            self._reinitialize()
            for _ in range(frame - 1):
                self.read(0)

    def read(self):
        self._current_frame += 1
        return self._cap.read()[1]



def surface_to_npim(surface):
    """ Transforms a Cairo surface into a numpy array. """
    im = +np.frombuffer(surface.get_data(), np.uint8)
    H,W = surface.get_height(), surface.get_width()
    im.shape = (H,W, 4) # for RGBA
    return im[:,:,:3]


def svg_to_npim(svg_bytestring, dpi=10):
    """ Renders a svg bytestring as a RGB image in a numpy array """
    tree = cairosvg.parser.Tree(bytestring=svg_bytestring)
    surf = cairosvg.surface.PNGSurface(tree,None, dpi).cairo
    return surface_to_npim(surf)


def board_arrays_to_mp4(board_arrays, file_name="file.avi"):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frame_rate = 29.96996465991079
    out = cv2.VideoWriter(file_name, fourcc, frame_rate, (400, 400))
    for svg in board_arrays_to_svgs(board_arrays):
        a = svg_to_npim(svg)
        out.write(a)
    out.release()


def show_together(file_name, board_arrays, start_at=0):
    v = VideoContainer(file_name)
    v.seek_to(start_at)

    i = start_at
    rendered = np.zeros((400,400))
    for svg in board_arrays_to_svgs(board_arrays)[start_at:]:

        if not (i % 4):
            rendered = svg_to_npim(svg)

        original = v.read()

        i += 1
        print(i)

        cv2.imshow('original', original)
        cv2.imshow('rendered', rendered)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break


