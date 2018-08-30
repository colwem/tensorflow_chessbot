import cv2
import numpy as np
from functools import reduce

#history 30 threshold 3 misses all events
#history 30 threshold 1 misses all events
#history 60 threshold 1 misses 2


def make_mask(vid, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    def get(v):
        subtractor = cv2.createBackgroundSubtractorMOG2(200, detectShadows=False)
        return [cv2.morphologyEx(subtractor.apply(frame), cv2.MORPH_OPEN, kernel)
                for frame in v]

    # kernel = np.ones((kernel_size, kernel_size), np.uint8)
    f = np.array(get(vid))
    b = np.array(list(reversed(get(reversed(vid)))))
    mask = np.asarray([cv2.bitwise_and(f[i], b[i]) for i in range(len(vid))])

    return f, b, mask


def find_stillness_events(squares, file, rank, min_event_len=2, post_event_len=1, threshold=1):

    n = squares.shape[0]

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(100, 16, detectShadows=False)
    event_window = []
    event_list = []
    num_frames_post_event = 0
    stillness_event_start = 0
    motion_event_start = 0
    in_motion_event = False

    # kernel_size = 1
    # kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Motion event scanning/detection loop.
    i = 0
    motion_score_max = 0
    motion_score_min = 100000000
    motion_event_start = None
    motion_event_end = None

    frame_width = 4
    for i in range(n):
        # bottom = squares[max(0, i - frame_width)]
        # top = squares[i]
        # score = np.sum(np.abs(top - bottom)) / (32 * 32)
        #
        frame_gray = squares[i]
        frame_filt = bg_subtractor.apply(frame_gray)
        if file == 2 and rank == 1 and i > 400:
            cv2.imshow('square', frame_gray.astype('uint8'))
            cv2.imshow('frame_filt', frame_filt.astype('uint8'))
            print(i)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break


        frame_score = np.sum(frame_filt) / float(frame_filt.shape[0] * frame_filt.shape[1])
        event_window.append(frame_score)
        event_window = event_window[-min_event_len:]

        if in_motion_event:
            # in event or post event, write all queued frames to file,
            # and write current frame to file.
            # if the current frame doesn't meet the threshold, increment
            # the current scene's post-event counter.
            if frame_score >= threshold:
                num_frames_post_event = 0
                motion_score_min = min(motion_score_min, frame_score)
                motion_score_max = max(motion_score_max, frame_score)
            else:
                num_frames_post_event += 1
                if num_frames_post_event >= post_event_len:
                    in_motion_event = False
                    stillness_event_start = i
                    motion_event_end = i

                    # print_motion_event(
                    #     motion_event_start,
                    #     motion_event_end,
                    #     motion_score_max,
                    #     motion_score_min)

        else:
            if len(event_window) >= min_event_len and all(
                    score >= threshold for score in event_window):
                motion_event_start = i
                motion_score_max = frame_score
                motion_score_min = frame_score
                stillness_event_end = i - min_event_len
                if stillness_event_end != stillness_event_start:
                    event_list.append((stillness_event_start, stillness_event_end))
                in_motion_event = True
                motion_event_start = i
                event_window = []
                num_frames_post_event = 0

    # If we're still in a motion event, we still need to compute the duration
    # and ending timecode and add it to the event list.
    if not in_motion_event:
        stillness_event_end = i
        event_list.append((stillness_event_start, stillness_event_end))

    return event_list


def print_motion_event( motion_event_start, motion_event_end, motion_score_max, motion_score_min):

    fm_str = '''
Motion Event

Start: frame {}, second {}
End:   frame {}, second {}

Max Score: {}
Min Score: {}
'''
    print(fm_str.format(
        motion_event_start,
        motion_event_start / 30,
        motion_event_end,
        motion_event_end / 30,
        motion_score_max,
        motion_score_min))

