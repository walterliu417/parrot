import re
import time
import math
import numpy as np
import io
import zstandard as zstd
import time
import json
import pickle
import matplotlib.pyplot as plt

from helperfuncs import *

print("imports complete")


your_filename = "D:/Parakeet/archives/lichess_db_eval.jsonl.zst"
dctx = zstd.ZstdDecompressor()
l = 0
num_datasets = 0
num_sub_datasets = 0
wlist, alist, dlist = [[], [], []], [[], [], []], [[], [], []]
wi, ai, di = 0, 0, 0
st = time.time()
i = 0
flag = None
with open(your_filename, 'rb') as compressed:
    with dctx.stream_reader(compressed) as reader:
        text_stream = io.TextIOWrapper(reader, encoding='utf-8')
        for line in text_stream:
            line = json.loads(line)
            try:
                cp = line["evals"][0]["pvs"][0]["cp"]
                score = cp_to_win_prob(cp)
                mate = None
                if 0.45 < score < 0.55:
                    # Draw: -0.5 ~< evaluation ~< +0.5
                    flag = position.DRAW
                elif 0.25 < score < 0.75:
                    # Advantage: -2.9 ~< evaluation ~< 2.9
                    flag = position.ADVANTAGE
                else:
                    # Completely winning position
                    flag = position.WINNING
            except:
                mate = line["evals"][0]["pvs"][0]["mate"]
                score = mate_to_win_prob(mate)
                flag = position.WINNING
            if flag == position.SKIP:
                l += 2
                continue
            else:
                fen = line["fen"]
                board = chess.Board(fen)
                if flag == position.WINNING:
                    wlist[0].append(fast_board_to_boardmap(board))
                    wlist[1].append(fast_board_to_feature(board))
                    wlist[2].append(score)
                elif flag == position.ADVANTAGE:
                    alist[0].append(fast_board_to_boardmap(board))
                    alist[1].append(fast_board_to_feature(board))
                    alist[2].append(score)
                elif flag == position.DRAW:
                    dlist[0].append(fast_board_to_boardmap(board))
                    dlist[1].append(fast_board_to_feature(board))
                    dlist[2].append(score)
                l += 1
                if len(wlist[2]) == 65536:
                    pickle.dump(wlist, open(f"D:/parakeet/evaluation_database/winning_data_{wi}.chess", "wb"))
                    print("win list,", wi)
                    wlist = [[], [], []]
                    wi += 1
                if len(alist[2]) == 65536:
                    pickle.dump(alist, open(f"D:/parakeet/evaluation_database/adv_data_{ai}.chess", "wb"))
                    print("adv list,", ai)
                    alist = [[], [], []]
                    ai += 1
                if len(dlist[2]) == 65536:
                    pickle.dump(dlist, open(f"D:/parakeet/evaluation_database/draw_data_{di}.chess", "wb"))
                    print("draw list,", di)
                    dlist = [[], [], []]
                    di += 1

                if l % 262144 == 262133:
                    print(l, time.time() - st)