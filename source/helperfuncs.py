import chess
import math
from enum import Enum
import numpy as np


DRAW_VALIDATION_SET = [25, 200, 300, 525]
ADV_VALIDATION_SET = [75, 200, 325, 450]
WIN_VALIDATION_SET = [50, 100, 225, 350]

class position(Enum):
    WINNING = 0
    ADVANTAGE = 1
    DRAW = 2
    SKIP = -1

chr_to_num = {"k": 0, "q": 1, "r": 2, "b": 3, "n": 4, "p": 5, "P": 7, "N": 8, "B": 9, "R": 10, "Q": 11, "K": 12}

def square_to_int(sq):
    return (ord(sq[0]) - 97) * 8 + int(sq[1]) - 1

def squareint_to_square(sqint):
    return (sqint // 8, sqint % 8)

def int_to_bin(anint, pad=4):
    return [int(_) for _ in "0" * (pad - len(bin(anint)[2:])) + bin(anint)[2:]]

def fast_board_to_boardmap(board):
    # Slower than piece_map() when there are less pieces on the board, but faster (~2x) in most cases.
    boards = [[0.5 for _ in range(8)] for _ in range(8)]
    for square in board.pieces(chess.PAWN, chess.WHITE):
        idx = squareint_to_square(square)
        boards[idx[0]][idx[1]] = round((chr_to_num["P"]) / 12, 4)
    for square in board.pieces(chess.PAWN, chess.BLACK):
        idx = squareint_to_square(square)
        boards[idx[0]][idx[1]] = round((chr_to_num["p"]) / 12, 4)
    for square in board.pieces(chess.KNIGHT, chess.WHITE):
        idx = squareint_to_square(square)
        boards[idx[0]][idx[1]] = round((chr_to_num["N"]) / 12, 4)
    for square in board.pieces(chess.KNIGHT, chess.BLACK):
        idx = squareint_to_square(square)
        boards[idx[0]][idx[1]] = round((chr_to_num["n"]) / 12, 4)
    for square in board.pieces(chess.BISHOP, chess.WHITE):
        idx = squareint_to_square(square)
        boards[idx[0]][idx[1]] = round((chr_to_num["B"]) / 12, 4)
    for square in board.pieces(chess.BISHOP, chess.BLACK):
        idx = squareint_to_square(square)
        boards[idx[0]][idx[1]] = round((chr_to_num["b"]) / 12, 4)
    for square in board.pieces(chess.ROOK, chess.WHITE):
        idx = squareint_to_square(square)
        boards[idx[0]][idx[1]] = round((chr_to_num["R"]) / 12, 4)
    for square in board.pieces(chess.ROOK, chess.BLACK):
        idx = squareint_to_square(square)
        boards[idx[0]][idx[1]] = round((chr_to_num["r"]) / 12, 4)
    for square in board.pieces(chess.QUEEN, chess.WHITE):
        idx = squareint_to_square(square)
        boards[idx[0]][idx[1]] = round((chr_to_num["Q"]) / 12, 4)
    for square in board.pieces(chess.QUEEN, chess.BLACK):
        idx = squareint_to_square(square)
        boards[idx[0]][idx[1]] = round((chr_to_num["q"]) / 12, 4)
    for square in board.pieces(chess.KING, chess.WHITE):
        idx = squareint_to_square(square)
        boards[idx[0]][idx[1]] = round((chr_to_num["K"]) / 12, 4)
    for square in board.pieces(chess.KING, chess.BLACK):
        idx = squareint_to_square(square)
        boards[idx[0]][idx[1]] = round((chr_to_num["k"]) / 12, 4)
    return [boards]

def fast_board_to_feature(board):
    whosemove = [int(board.turn)]
    enpassqnum = board.ep_square
    can_enpassant = [0]
    if (enpassqnum is not None) and (board.has_legal_en_passant()):
        enpassqnum = (enpassqnum % 8) * 8 + (enpassqnum // 8)
        enpassqnum = int_to_bin(enpassqnum, pad=6)
        can_enpassant = [1]
    else:
        enpassqnum = int_to_bin(0, pad=6)
    castling_rights = [int(board.has_kingside_castling_rights(chess.WHITE)), int(board.has_queenside_castling_rights(chess.WHITE)), int(board.has_kingside_castling_rights(chess.BLACK)), int(board.has_queenside_castling_rights(chess.BLACK))]
    return whosemove + can_enpassant + enpassqnum + castling_rights

def cp_to_win_prob(cp):
    return 0.4 * (1 - math.exp(-cp / 200)) / (1 + math.exp(-cp / 200)) + 0.5

def mate_to_win_prob(mate):
    if mate < 0:
        return min(0.1, 0.1 + (abs(mate) - 21)/200)
    else:
        return max(0.9, 0.9 + (21 - mate)/200)
