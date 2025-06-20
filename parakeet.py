import numpy as np
import chess
import helperfuncs
from helperfuncs import *
from search import *

class Parakeet:

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_remaining = 0
        self.root_node = None 
        sess_options = onnxruntime.SessionOptions()
        sess_options.intra_op_num_threads = helperfuncs.num_cores
        self.model = onnxruntime.InferenceSession(helperfuncs.model_path, sess_options, providers=[helperfuncs.provider])
        if helperfuncs.log: print("Current best model loaded successfully!")

    def set_fen(self, fen):
        self.board = chess.Board(fen)

    def search(self, movetime, wtime, btime):
        start = time.time()
        if helperfuncs.broken:
            helperfuncs.broken = False
            sess_options = onnxruntime.SessionOptions()

            self.model = onnxruntime.InferenceSession(helperfuncs.model_path, sess_options, providers=[helperfuncs.provider])
        helperfuncs.nodes = 0
        
        if not self.root_node:
            self.root_node = Node(self.board, self.model, None, None)
        else:
            tt = False
            for child in self.root_node.children:
                if child.board.fen() == self.board.fen():
                    tt = True
                    self.root_node = child
                    break
            if not tt: self.root_node = Node(self.board, self.model, None, None)
        self.root_node.parent = None # Clear memory used by previous node
        
        if movetime > 0:
            self.time_for_this_move = movetime
        else:
            # Simple time management
            if self.board.turn == chess.WHITE:
                self.time_remaining = wtime
            elif self.board.turn == chess.BLACK:
                self.time_remaining = btime
                
            if self.board.fullmove_number < 5:
                # Opening - save time
                self.time_for_this_move = self.time_remaining * 0.4 / 25
            elif self.board.fullmove_number < 30:
                # Midgame - use time
                self.time_for_this_move = self.time_remaining * 0.8 / 25
            else:
                # Endgame - save time so there are no blunders later on!
                self.time_for_this_move = (self.time_remaining / 40)

        child = self.root_node.pns(time.time(), self.time_for_this_move)

        movelist = []
        bestmove = child.move
        cp = nn_to_cp(self.root_node.value)
        if helperfuncs.datagen: data_boards = [self.root_node.board]
        while (child.children != []) and not child.terminal:
            if helperfuncs.datagen: data_boards.append(child.board)
            movelist.append(child.move.uci())
            child = min(child.children, key=lambda c: c.value)
        nps = helperfuncs.nodes / self.time_for_this_move
        if len(movelist) != 0:
            if helperfuncs.log: print(f"info depth 1 seldepth {len(movelist)} time {int((time.time() - start) * 1000)} nodes {helperfuncs.nodes} score cp {int(cp * 100)} nps {int(nps)} pv {' '.join(movelist)}")
        else:
            if helperfuncs.log: print(f"info depth 1 seldepth 1 time {int((time.time() - start) * 1000)} nodes {helperfuncs.nodes} score cp {int(cp * 100)} nps {int(nps)} pv {bestmove}")

        if helperfuncs.datagen:
            board_list = []
            eval_list = []
            max_visits = max(self.root_node.children, key=lambda c: c.visits)
            for c in self.root_node.children:
                if c.visits > max_visits.visits / 2 and c.move != bestmove:
                    data_boards.append(c.board)
            for b in data_boards:
                board_list.append(fast_board_to_boardmap(b))
                value = stockfish_analyse(b)
                eval_list.append(value)
            return bestmove, board_list, eval_list

        return bestmove


def run():
    while True:
        command = input()
        if command == "":
            continue
        command = command.split()        
        if command[0] == "uci":
            print("id name Parakeet v1.2")
            print("id author Walter Liu")
            print("option name explore_factor type spin default 20 min 0 max 200")
            print("option name capture_bonus type spin default 350 min 0 max 500")
            print("option name check_bonus type spin default 100 min 0 max 500")
            print("option name explore_decay type spin default 100 min 0 max 500")
            print("option name tablebase_dir type string default /content/drive/MyDrive/parakeet/tablebase_5pc")
            print("option name net_path type string default parakeet.onnx")
            print("option name gpu_enabled type check default true")
            print("option name num_threads type spin default 1 min 0 max 64")
            print("option name temperature type spin default 0 min 0 max 100")
            print("option name temp_moves type spin default 5 min 0 max 100")
            print("option name datagen type check default false")
            print("option name log type check default true")
            print("uciok")
        elif command[0] == "isready":
            print("readyok")
        elif command[0] == "ucinewgame":
            engine = Parakeet()
        elif command[0] == "quit":
            import sys
            sys.exit()
        elif command[0] == "position":
            if command[1] == "fen":
                engine.set_fen(' '.join(command[2:]))

            elif command[1] == "startpos":
                engine.set_fen(chess.STARTING_FEN)
                if len(command) > 2:
                    if command[2] == "moves":
                        engine.set_fen(chess.STARTING_FEN)
                        for move in command[3:]:
                            engine.board.push_uci(move)
        elif command[0] == "go":  # TODO: command parsing needs to be redone
            movetime = 0
            wtime = 0
            btime = 0

            if command[1] == "movetime":
                movetime = float(command[2]) / 1000.0
            if command[1] == "wtime":
                wtime = float(command[2]) / 1000.0
            if command[1] == "btime":
                btime = float(command[2]) / 1000.0
            if len(command) > 3 and command[3] == "btime":
                btime = float(command[4]) / 1000.0
            if len(command) > 5 and command[5] == "btime":
                btime = float(command[6]) / 1000.0

            if helperfuncs.log: print(f"bestmove {engine.search(movetime, wtime, btime)}")
        elif command[0] == "setoption":
            name = command[2]
            if name == "explore_factor":
                helperfuncs.factor = float(command[4]) /  100.0
            elif name == "capture_bonus":
                helperfuncs.quiescent = float(command[4]) / 1000.0
            elif name == "check_bonus":
                helperfuncs.check = float(command[4]) / 1000.0
            elif name == "explore_decay":
                helperfuncs.decay = float(command[4]) / 100.0
            elif name == "tablebase_dir":
                try:
                    helperfuncs.TABLEBASE = chess.syzygy.open_tablebase(command[4])
                    print(f"Tablebase found at {command[4]}.")
                except:
                    print("Tablebase not found!")
            elif name == "net_path":
                helperfuncs.model_path = command[4]
            elif name == "gpu_enabled":
                if command[4] == "true":
                    helperfuncs.provider = "CUDAExecutionProvider"
                elif command[4] == "false":
                    helperfuncs.provider = "CPUExecutionProvider"
            elif name == "num_threads":
                helperfuncs.num_cores = int(command[4])
            elif name == "temperature":
                helperfuncs.temperature = int(command[4])
            elif name == "temp_moves":
                helperfuncs.temp_moves = int(command[4])
            elif name == "datagen":
                if command[4] == "true":
                    helperfuncs.datagen = True
                elif command[4] == "false":
                    helperfuncs.datagen = False
            elif name == "log":
                if command[4] == "true":
                    helperfuncs.log = True
                elif command[4] == "false":
                    helperfuncs.log = False

if __name__ == "__main__":
    run()
