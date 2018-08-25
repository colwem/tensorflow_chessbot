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
import csv

from helper_functions import shortenFEN
from  helper_image_loading import loadImageFromPath as load_image_from_path
from chessboard_finder import findChessboardCorners as find_chessboard_corners, getChessTilesGray as get_chess_tiles_gray
from helper_functions import split_by_fun

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


def dedup(imgs, lst):
    return imgs


def get_tiles(img, corners):
    return get_chess_tiles_gray(img, corners)

def piece_list_to_fen(lst):
    if len(lst) != 64:
        raise Exception("length of piece list is not 64")
    return '/'.join([''.join(lst[i * 8:(i + 1) * 8]) for i in reversed(range(8))])

###########################################################
# MAIN CLI

def main(args):
    # Load image from filepath or URL
    # Initialize predictor, takes a while, but only needed once
    with open('fens.csv', 'w') as fenCsvFile:
        writer = csv.writer(fenCsvFile, dialect='excel')
        predictor = ChessboardPredictor()
        # Load image from file
        # open dir, loop through files

        for path_list in chunk(os.listdir(args.dirpath), 100):

            img_list = [load_image_from_path(os.path.join(args.dirpath, path)) for path in path_list]
            img_list = [np.asarray(img.convert("L"), dtype=np.float32) for img in img_list]

            corner_groups = split_by_fun(img_list, find_chessboard_corners, lambda x, y: (x == y).all(), mindepth=4)

            tiled = [get_tiles(img, corners) for imgs, corners in corner_groups for img in imgs]
            tiles = np.concatenate(tiled, 2)
            # tiled, path_list = dedup(tiled, path_list)

            # Resize image if too large
            # img = helper_image_loading.resizeAsNeeded(img)

            pieces, certainties = predictor.get_predictions(tiles)
            fens = map(lambda l: shortenFEN(piece_list_to_fen(l)), chunk(pieces, 64))
            certainties = [c.min() for c in chunk(certainties, 64)]
            table = zip(fens, certainties, path_list)
            print(list(table))
            # short_fens = [shortenFEN(fen) for fen in fens]
            # # Use the worst case certainty as our final uncertainty score
            # certainty = tile_certainties.min()
            # # writer.writerow(['', short_fen, certainty, fn])
            # print(short_fens)
            # print('.', end='', flush=True)


    predictor.close()

if __name__ == '__main__':
    np.set_printoptions(suppress=True, precision=3)
    import argparse

    parser = argparse.ArgumentParser(description='Predict a chessboard FENs from supplied directory of local images')
    parser.add_argument('--dirpath', required=True, help='dirpath to find images to process')
    args = parser.parse_args()
    main(args)
