#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# TensorFlow Chessbot
# This contains ChessboardPredictor, the class responsible for loading and
# running a trained CNN on chessboard screenshots. Used by chessbot.py.
# A CLI interface is provided as well.
#
#   $ ./tensorflow_chessbot.py -h
#   usage: tensorflow_chessbot.py [-h] [--url URL] [--filepath FILEPATH]
#
#    Predict a chessboard FEN from supplied local image link or URL
#
#    optional arguments:
#      -h, --help           show this help message and exit
#      --url URL            URL of image (ex. http://imgur.com/u4zF5Hj.png)
#     --filepath FILEPATH  filepath to image (ex. u4zF5Hj.png)
#
# This file is used by chessbot.py, a Reddit bot that listens on /r/chess for
# posts with an image in it (perhaps checking also for a statement
# "white/black to play" and an image link)
#
# It then takes the image, uses some CV to find a chessboard on it, splits it up
# into a set of images of squares. These are the inputs to the tensorflow CNN
# which will return probability of which piece is on it (or empty)
#
# Dataset will include chessboard squares from chess.com, lichess
# Different styles of each, all the pieces
#
# Generate synthetic data via added noise:
#  * change in coloration
#  * highlighting
#  * occlusion from lines etc.
#
# Take most probable set from TF response, use that to generate a FEN of the
# board, and bot comments on thread with FEN and link to lichess analysis.
#
# A lot of tensorflow code here is heavily adopted from the
# [tensorflow tutorials](https://www.tensorflow.org/versions/0.6.0/tutorials/pdes/index.html)

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Ignore Tensorflow INFO debug messages
import tensorflow as tf
import numpy as np
import cv2
import csv

from helper_functions import shortenFEN
from chessboard_finder import findChessboardCorners as find_chessboard_corners, getChessTilesGray as get_chess_tiles_gray
from helper_functions import split_by_fun
from helper_video import get_video_array

def load_graph(frozen_graph_filepath):
    # Load and parse the protobuf file to retrieve the unserialized graph_def.
    with tf.gfile.GFile(frozen_graph_filepath, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Import graph def and return.
    with tf.Graph().as_default() as graph:
        # Prefix every op/nodes in the graph.
        tf.import_graph_def(graph_def, name="tcb")
    return graph


class ChessboardPredictor(object):
    """ChessboardPredictor using saved model"""

    def __init__(self, frozen_graph_path='saved_models/frozen_graph.pb'):
        # Restore model using a frozen graph.
        print("\t Loading model '%s'" % frozen_graph_path)
        graph = load_graph(frozen_graph_path)
        self.sess = tf.Session(graph=graph)

        # Connect input/output pipes to model.
        self.x = graph.get_tensor_by_name('tcb/Input:0')
        self.keep_prob = graph.get_tensor_by_name('tcb/KeepProb:0')
        self.prediction = graph.get_tensor_by_name('tcb/prediction:0')
        self.probabilities = graph.get_tensor_by_name('tcb/probabilities:0')
        print("\t Model restored.")

    def get_predictions(self, tiles):
        """Run trained neural network on tiles generated from image"""
        if tiles is None or len(tiles) == 0:
            print("Couldn't parse chessboard")
            return None, 0.0

        # Reshape into Nx1024 rows of input data, format used by neural network
        N = tiles.shape[2]
        validation_set = np.swapaxes(np.reshape(tiles, [32 * 32, N]), 0, 1)

        # Run neural network on data
        guess_prob, guessed = self.sess.run(
            [self.probabilities, self.prediction],
            feed_dict={self.x: validation_set, self.keep_prob: 1.0})

        # Prediction bounds
        a = np.array(list(map(lambda x: x[0][x[1]], zip(guess_prob, guessed))))
        labelIndex2Name = lambda label_index: ' KQRBNPkqrbnp'[label_index]
        pieceNames = list(map(lambda k: '1' if k == 0 else labelIndex2Name(k), guessed))  # exchange ' ' for '1' for FEN
        return pieceNames, a


        # tile_certainties = a.reshape([8, 8])[::-1, :]
        #
        # # Convert guess into FEN string
        # # guessed is tiles A1-H8 rank-order, so to make a FEN we just need to flip the files from 1-8 to 8-1
        # labelIndex2Name = lambda label_index: ' KQRBNPkqrbnp'[label_index]
        # pieceNames = list(map(lambda k: '1' if k == 0 else labelIndex2Name(k), guessed))  # exchange ' ' for '1' for FEN
        # fen = '/'.join([''.join(pieceNames[i * 8:(i + 1) * 8]) for i in reversed(range(8))])

    def close(self):
        print("Closing session.")
        self.sess.close()


def chunk(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i + n]


def is_dup(a, b):
    c = a - b
    max_ave_diff = np.max(np.apply_over_axes(np.average, c, [0, 1]))
    return max_ave_diff < 0.001

def convert_to_chessboard(vid, corner_group):
    # img is a grayscale image
    # corners = (x0, y0, x1, y1) for top-left corner to bot-right corner of board
    start, end = corner_group[0]
    length = vid.shape[0]

    corners = corner_group[1]

    height, width = vid.shape[1:3]

    # corners could be outside image bounds, pad image as needed
    padl_x = max(0, -corners[0])
    padl_y = max(0, -corners[1])
    padr_x = max(0, corners[2] - width)
    padr_y = max(0, corners[3] - height)

    vid_padded = np.pad(vid, ((0, 0), (padl_y, padr_y), (padl_x, padr_x)), mode='edge')

    chessboard_vid = vid_padded[:,
    (padl_y + corners[1]):(padl_y + corners[3]),
    (padl_x + corners[0]):(padl_x + corners[2])]

    # 256x256 px image, 32x32px individual tiles
    # Normalized

    resized = np.zeros((length, 256, 256))
    for i in range(start, end):
        resized[i] = cv2.resize(chessboard_vid[i], (256, 256))

    return resized

def get_tiles(img, corners):
    return get_chess_tiles_gray(img, corners)

def piece_list_to_fen(lst):
    if len(lst) != 64:
        raise Exception("length of piece list is not 64")
    return '/'.join([''.join(lst[i * 8:(i + 1) * 8]) for i in reversed(range(8))])


def get_corner_groups(vid):
    # find chessboard corners
    def eqlf(a, b):
        if a is None or b is None:
            if a is None and b is None:
                return True
            return False
        return (a == b).all()

    corner_groups = split_by_fun(vid, find_chessboard_corners, eqlf, mindepth=0)
    print('len cg bf ', len(corner_groups))
    print('corners ', corner_groups[0][1])

    corner_groups = list(filter(lambda x: x[1] is not None, corner_groups))
    print('len cg af ', len(corner_groups))
    return corner_groups


def convert_to_tiles(vid):
    # shape (N, height, width) -> (N, file, rank, height, width)
    # shape (N,    256,   256) -> (N,    8,    8,     32,    32)
    N = vid.shape[0]
    r = np.zeros((N, 8, 8, 32, 32))
    for i in range(N):
        for j in range(8):
            for k in range(8):
                h_start = (32 * j)
                h_end = (32 * (j + 1))
                w_start = (32 * k)
                w_end = (32 * (k + 1))
                r[i,j,k] = vid[i, h_start:h_end, w_start:w_end]
    return r

def find_motion_events(vid):

    N = vid.shape[0]

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    event_window = []
    event_list = []
    num_frames_post_event = 0
    event_start = None

    kernel_size = 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    min_event_len = 5
    post_event_len = 2
    threshold = 0.15

    # curr_state = 'no_event'     # 'no_event', 'in_event', or 'post_even
    in_stillness_event = False


    # Motion event scanning/detection loop.
    for i in range(N):
        frame_gray = vid[i]

        frame_filt = bg_subtractor.apply(frame_gray)
        frame_score = np.sum(frame_filt) / float(frame_filt.shape[0] * frame_filt.shape[1])
        event_window.append(frame_score)
        event_window = event_window[-min_event_len:]

        if in_stillness_event:
            # in event or post event, write all queued frames to file,
            # and write current frame to file.
            # if the current frame doesn't meet the threshold, increment
            # the current scene's post-event counter.
            if frame_score <= threshold:
                num_frames_post_event = 0
            else:
                num_frames_post_event += 1
                if num_frames_post_event >= post_event_len:
                    in_stillness_event = False
                    event_end = i
                    event_duration = i - event_start
                    event_list.append((event_start, event_end, event_duration))
        else:
            if len(event_window) >= min_event_len and all(
                    score <= threshold for score in event_window):
                in_stillness_event = True
                event_window = []
                num_frames_post_event = 0
                event_start = i

    # If we're still in a motion event, we still need to compute the duration
    # and ending timecode and add it to the event list.
    if in_stillness_event:
        event_end = i
        event_duration = i - event_start
        event_list.append((event_start, event_end, event_duration))

    return event_list

###########################################################
# MAIN CLI

def main(args):
    # Load image from filepath or URL
    # Initialize predictor, takes a while, but only needed once
    predictor = ChessboardPredictor()
    # Load image from file
    # open dir, loop through files
    table = []

    for vid, frames in get_video_array(args.filepath, 500):

        corner_groups = get_corner_groups(vid)

        if len(corner_groups):
            # extract tiles, returns N elements of 64 tiles
            for corner_group in corner_groups:
                board_vid = convert_to_chessboard(vid, corner_group)
                tile_vid = convert_to_tiles(board_vid)
                events = [[find_motion_events(tile_vid[:, i, j, :, :]) for j in range(8)] for i in range(8)]
                print(events)

            exit()
            tiled = [get_tiles(img, corners) for imgs, corners in corner_groups for img in imgs]
            print('len tl bf ', len(tiled))

            # dedup tiles
            current = 0
            new_tiled = [tiled[0]]
            new_frames = [frames[0]]
            for i in range(1, len(tiled) - 1):
                if not is_dup(tiled[current], tiled[i]):
                    current = i
                    new_tiled.append(tiled[current])
                    new_frames.append(frames[current])
            tiled, frames = new_tiled, new_frames
            print('len tl af ', len(tiled))

            if len(tiled):
                # tiled, _, path_list =\
                #     zip((tiled[0], 0, path_list[0]),\
                #         *filter(lambda x: not is_dup(x[0], x[1]), zip(tiled, tiled[1:], path_list[1:])))

                tiles = np.concatenate(tiled, 2)
                # tiled, path_list = dedup(tiled, path_list)

                # predictions
                pieces, certainties = predictor.get_predictions(tiles)

                # convert predictions into fens
                fens = map(lambda l: shortenFEN(piece_list_to_fen(l)), chunk(pieces, 64))

                # Use the worst case certainty as our final uncertainty score
                certainties = [c.min() for c in chunk(certainties, 64)]

                # write fen, certainty and file paths to csv file
                table.extend(zip(fens, certainties, frames))

    # from itertools import groupby
    # print('l tab bf ', len(table))
    # table = [list(g)[0] for k, g in groupby(table, lambda x: x[0])]
    # print('l tab af ', len(table))

    with open('fens.csv', 'w') as fenCsvFile:
        writer = csv.writer(fenCsvFile, dialect='excel')
        writer.writerows(table)


    predictor.close()

if __name__ == '__main__':
    np.set_printoptions(suppress=True, precision=3)
    import argparse

    parser = argparse.ArgumentParser(description='Predict a chessboard FENs from supplied directory of local images')
    parser.add_argument('-f', '--filepath', required=True, help='path to video to process')
    args = parser.parse_args()
    main(args)

