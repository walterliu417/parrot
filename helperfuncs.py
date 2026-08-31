import time
import chess
import chess.syzygy
import chess.engine
import numpy as np
import random
import onnxruntime
import math
import sys
import os
from pathlib import Path

cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"

onnxruntime.preload_dlls(
    cuda=True,
    cudnn=False,
    msvc=False,
    directory=cuda_bin
)

cudnn_bin = r"C:\Program Files\NVIDIA\CUDNN\v9.24\bin\12.9\x64"

onnxruntime.preload_dlls(
    cuda=False,
    cudnn=True,
    msvc=False,
    directory=cudnn_bin
)

# Global variables
nodes = 0
factor = 0.2
decay = 1
quiescent = 0
check = 0.1
temperature = 0
temp_moves = 5
model_path = "parakeet.onnx" # Around 5x speedup on GPU.
broken = False
provider = "CUDAExecutionProvider" # default: gpu enabled.
num_cores = 1
datagen = False
log = True

teacher = None

# Look for Syzygy tablebase
try:
    TABLEBASE = chess.syzygy.open_tablebase("/content/drive/MyDrive/parakeet/tablebase_5pc")
    print("5 piece Syzygy endgame tablebase found.")
except:
    print("Could not find tablebase")
    TABLEBASE = None

chr_to_num = {
    "k": 0,
    "q": 1,
    "r": 2,
    "b": 3,
    "n": 4,
    "p": 5,
    "P": 7,
    "N": 8,
    "B": 9,
    "R": 10,
    "Q": 11,
    "K": 12,
}


def application_dir():
    if getattr(sys, "frozen", False):
        # Directory containing parakeet.exe
        return Path(sys.executable).resolve().parent

    # Directory containing helperfuncs.py during normal Python execution
    return Path(__file__).resolve().parent


def create_teacher():
    base = application_dir()

    if sys.platform == "win32":
        stockfish_path = (
            base
            / "engines"
            / "stockfish-windows"
            / "stockfish-windows-x86-64-avx2.exe"
        )
    elif sys.platform == "linux":
        stockfish_path = (
            base
            / "engines"
            / "stockfish_14_linux_x64_avx2"
            / "stockfish_14_x64_avx2"
        )
    else:
        raise RuntimeError(
            f"Unsupported platform for Stockfish: {sys.platform}"
        )

    if not stockfish_path.exists():
        raise FileNotFoundError(
            f"Stockfish not found at: {stockfish_path}"
        )

    return chess.engine.SimpleEngine.popen_uci(
        str(stockfish_path)
    )

def square_to_int(sq):
    return (ord(sq[0]) - 97) * 8 + int(sq[1]) - 1

def squareint_to_square(sqint):
    return (sqint // 8, sqint % 8)

def int_to_bin(anint, pad=4):
    return [int(_) for _ in "0" * (pad - len(bin(anint)[2:])) + bin(anint)[2:]]

def nn_to_cp(score):
    if 0.1 < score < 0.9:
        return -2 * np.log((0.9 - score) / (score - 0.1))
    elif score <= 0.1:
        return -100
    elif score >= 0.9:
        return 100

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

def lt5(board):
    # Are there less then 5 pieces on the board? If so, go to tablebase probing.
    p = 0
    p += len(board.pieces(chess.PAWN, chess.WHITE))
    if p > 3: return False
    p += len(board.pieces(chess.PAWN, chess.BLACK))
    if p > 3: return False
    p += len(board.pieces(chess.KNIGHT, chess.WHITE))
    p += len(board.pieces(chess.KNIGHT, chess.BLACK))
    p += len(board.pieces(chess.BISHOP, chess.WHITE))
    p += len(board.pieces(chess.BISHOP, chess.BLACK))
    p += len(board.pieces(chess.ROOK, chess.WHITE))
    if p > 3: return False
    p += len(board.pieces(chess.ROOK, chess.BLACK))
    if p > 3: return False
    p += len(board.pieces(chess.QUEEN, chess.WHITE))
    p += len(board.pieces(chess.QUEEN, chess.BLACK))
    return p <= 3

def cp_to_win_prob(cp):
    return 0.4 * (1 - math.exp(-cp / 200)) / (1 + math.exp(-cp / 200)) + 0.5

def mate_to_win_prob(mate):
    if mate < 0:
        return min(0.1, 0.1 + (abs(mate) - 21)/200)
    else:
        return max(0.9, 0.9 + (21 - mate)/200)
    
def stockfish_analyse(board, time=0.5):
    global teacher

    if teacher is None:
        teacher = create_teacher()

    info = teacher.analyse(
        board,
        chess.engine.Limit(time=time)
    )["score"]

    score = info.pov(chess.WHITE)

    if score.is_mate():
        return mate_to_win_prob(score.mate())

    return cp_to_win_prob(score.score())