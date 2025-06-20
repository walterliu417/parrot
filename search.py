import helperfuncs
from helperfuncs import *
from typing import Optional

class Node:
    pass
class Node:

    def __init__(self, board: chess.Board, net: onnxruntime.InferenceSession, move: Optional[chess.Move]=None, parent: Optional[Node]=None, depth=0):
        self.board = board
        self.move = move
        self.value = None
        self.parent = parent
        self.visits = 0
        self.depth = depth

        self.net = net
        self.children = []
        self.flag = None
        self.terminal = False

        try:
            self.capture = self.parent.board.is_capture(self.move)
        except:
            self.capture = False

        try:
            self.check = self.parent.board.is_check(self.move)
        except:
            self.check = False

        try:
            if self.move.promotion is not None:
                self.promotion = True
            else:
                self.promotion = False
        except:
            self.promotion = False


    def ucb(self, time_fraction):
        bonus = 0
        if self.capture:
            bonus = helperfuncs.quiescent
        elif self.check:
            bonus = helperfuncs.check
        elif self.promotion:
            bonus = helperfuncs.quiescent
        return self.value - helperfuncs.factor * (1 - (min(helperfuncs.decay, 1) * time_fraction)) * np.sqrt(np.log(self.parent.visits + 1) / (self.visits + 1)) - (bonus * helperfuncs.factor * (1 - min(helperfuncs.decay, 1) * time_fraction))

    def evaluate_nn(self):
        boardlist = fast_board_to_boardmap(self.board)
        if not self.board.turn:
            boardlist = np.rot90(boardlist, 2) * -1
            boardlist = boardlist.tolist()
        pos = np.array(boardlist).astype(np.float32).reshape(1, 1, 8, 8)
        ort_inputs = {"input": pos}
        return self.net.run(None, ort_inputs)[0]

    def evaluate_position(self):
        # Avoid threefold, since tablebases don't check for it for some reason.
        outcome = self.board.result(claim_draw=True)
        if outcome != "*":
            if (outcome == "1-0") and self.board.turn:
                return 2 + max((10 - self.depth) / 10, 0)
            elif (outcome == "0-1") and self.board.turn:
                return -1 - max((10 - self.depth) / 10, 0)
            elif (outcome == "1-0") and not self.board.turn:
                return -1 - max((10 - self.depth) / 10, 0)
            elif (outcome == "0-1") and not self.board.turn:
                return 2 + max((10 - self.depth) / 10, 0)
            elif outcome == "1/2-1/2":
                return 0.5
        if helperfuncs.TABLEBASE and lt5(self.board):
            result = helperfuncs.TABLEBASE.probe_dtz(self.board)
            if 1 <= result <= 100:
                return 1 + (100 - result) / 100
            elif -100 <= result <= -1:
                return 0 - (100 + result) / 100
            elif result == 0 or (result < -100) or (result > 100):
                return 0.5
        return None

    def generate_children(self):
        all_positions = []

        evaled = []
        not_evaled = []
        blm = self.board.legal_moves
        helperfuncs.nodes += blm.count()
        for move in blm:
            newboard = self.board.copy()
            newboard.push(move)
            newnode = Node(newboard, self.net, move, self, depth=self.depth + 1)
            if newnode.board.halfmove_clock > 100:
                newnode.value = 0.5
                newnode.terminal = True
                evaled.append(newnode)
            else:
                score = newnode.evaluate_position()
                if score is not None:
                    newnode.value = score
                    newnode.terminal = True
                    evaled.append(newnode)
                else:
                    boardlist = fast_board_to_boardmap(newboard)
                    if not newboard.turn:
                        boardlist = np.rot90(boardlist, 2) * -1
                        boardlist = boardlist.tolist()
                    all_positions.append(boardlist)
                    not_evaled.append(newnode)

        if len(not_evaled) > 0:
            pos = np.array(all_positions).astype(np.float32).reshape(len(not_evaled), 1, 8, 8)
            ort_inputs = {"input": pos}

            while True:
                try:
                    # Weird BatchNorm error that happens sometimes?? Attempt to recover.
                    result = self.net.run(None, ort_inputs)[0]
                    break
                except:
                    sess_options = onnxruntime.SessionOptions()
                    sess_options.intra_op_num_threads = helperfuncs.num_cores
                    self.net = onnxruntime.InferenceSession(helperfuncs.model_path, sess_options, providers=[helperfuncs.provider])
                    helperfuncs.broken = True

        for i in range(len(not_evaled)):
            if not_evaled[i].board.fullmove_number <= helperfuncs.temp_moves:
                not_evaled[i].value = float(result[i]) * (1 + random.random() * helperfuncs.temperature / 100)
            else:
                not_evaled[i].value = float(result[i])
            evaled.append(not_evaled[i])

        self.children = evaled


    def pns(self, start_time, time_for_this_move) -> Node:
        if helperfuncs.log: print(f"info string explore_factor {helperfuncs.factor} capture_bonus {helperfuncs.quiescent} check_bonus {helperfuncs.check} explore_decay {helperfuncs.decay}")
        while time.time() - start_time < time_for_this_move:
            # 1. Traverse tree with UCT + quiescence and decreasing exploration with time.
            target_node = self
            while target_node.children != []:
                target_node.visits += 1
                time_fraction = (time.time() - start_time) / time_for_this_move
                target_node = min(target_node.children, key=lambda child: child.ucb(time_fraction))

            # 2. Expansion and simulation
            target_node.generate_children()
            target_node.visits += 1
            score = target_node.evaluate_position()
            if score is not None:
                target_node.terminal = True
                target_node.value = score

            if not target_node.terminal:

                # 3. Backpropagation
                while True:
                    if target_node.children == []:
                        target_node.value = target_node.evaluate_position()
                        if target_node.value is None:
                            target_node.value = target_node.evaluate_nn()
                    else:
                        best_child_value = 1 - min(target_node.children, key=lambda child: child.value).value
                        if target_node.value is None:
                            target_node.value = best_child_value
                        else:
                            target_node.value = target_node.value * 0.75 + best_child_value * 0.25 # Conservative update.
                    if target_node.parent is not None:
                        target_node = target_node.parent
                    else:
                        break

        # 4. Select move - UBFMS
        max_visits = max(self.children, key=lambda child: child.visits)
        selected_child = min(self.children, key=lambda child: child.value)
        if helperfuncs.log: print(f"info string root_visits {self.visits} max_visits {max_visits.visits} best_visits {selected_child.visits}")
        options = []
        for child in self.children:
            if child.visits == max_visits.visits:
                options.append(child)
        if (selected_child.value >= 1):
            # Return best tablebase move immediately.
            return selected_child
        elif (selected_child.value > 0.7) and (selected_child.visits > 1):
            # Attempt to force a win?
            return selected_child
        else:
            return min(options, key=lambda c: c.value)
