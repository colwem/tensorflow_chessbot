import chess
import chess.svg
from functools import reduce


def board_arrays_to_svgs(board_arrays):
    return [board_array_to_svg(board) for board in board_arrays]


def board_arrays_to_fens(board_arrays):
    return [fen_from_board_array(board_array) for board_array in board_arrays]


def board_array_to_svg(board_array):
    board = chess.Board()
    board.set_board_fen(fen_from_board_array(board_array))
    return chess.svg.board(board)


def fen_from_board_array(board_array, file_axis=0, rank_axis=1, key=' KQRBNPkqrbnp'):
    board_array = board_array.copy()
    board_array[board_array > 12] = 0
    fen = '/'.join([''.join([key[piece] for piece in row]) for row in board_array.swapaxes(0,1)])
    return shorten_fen(fen)


def shorten_fen(fen):
    replace_char = ' '
    return reduce(lambda s, x: s.replace(*x), [(replace_char * i, str(i)) for i in reversed(range(1,9))], fen)


def piece_list_to_fen(lst):
    if len(lst) != 64:
        raise Exception("length of piece list is not 64")
    return '/'.join([''.join(lst[i * 8:(i + 1) * 8]) for i in reversed(range(8))])
