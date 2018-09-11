from chess_helpers import board_to_board_array
import numpy as np
import chess


def test_board_to_board_array():
    board = chess.Board()
    wanted = np.array( [[9, 11, 10, 8, 7, 10, 11, 9],
                        [12, 12, 12, 12, 12, 12, 12, 12],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [6, 6, 6, 6, 6, 6, 6, 6],
                        [3, 5, 4, 2, 1, 4, 5, 3]])
    assert (board_to_board_array(board) == wanted).all()
