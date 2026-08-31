import helperfuncs
from helperfuncs import *
from typing import Optional


class Node:
    """
    Search node using a fixed WHITE-perspective evaluation convention:

        larger value = better for White
        smaller value = better for Black
        0.5          = approximately equal

    Two values are intentionally maintained:

        value
            EMA-smoothed value used for tree traversal / exploration.  This
            damps noisy neural-network leaf evaluations.

        minimax_value
            Current unsmoothed minimax value of the already-expanded tree.
            This propagates new information all the way toward the root in a
            single backup pass instead of passing through an EMA at every ply.

    Positions are fed to the network exactly as fast_board_to_boardmap()
    produces them.  There is no rotate/negate transform for Black-to-move
    positions; side to move is supplied explicitly as input plane 1.
    """

    def __init__(
        self,
        board: chess.Board,
        net: onnxruntime.InferenceSession,
        move: Optional[chess.Move] = None,
        parent: Optional["Node"] = None,
        depth: int = 0,
    ):
        self.board = board
        self.move = move

        # Smoothed value used for traversal.
        self.value = None

        # Fresh current minimax value used for backup/final choice.
        self.minimax_value = None

        self.parent = parent
        self.visits = 0
        self.depth = depth

        self.net = net
        self.children = []
        self.flag = None
        self.terminal = False

        if self.parent is not None and self.move is not None:
            self.capture = self.parent.board.is_capture(self.move)
            # self.board is the position AFTER self.move.
            self.check = self.board.is_check()
            self.promotion = self.move.promotion is not None
        else:
            self.capture = False
            self.check = False
            self.promotion = False

    # ------------------------------------------------------------------
    # Search scoring
    # ------------------------------------------------------------------
    def tactical_bonus(self):
        if self.capture:
            return helperfuncs.quiescent
        if self.check:
            return helperfuncs.check
        if self.promotion:
            return helperfuncs.quiescent
        return 0.0

    def exploration_amount(self, time_fraction):
        """Return a positive exploration amount.

        White adds it and maximizes; Black subtracts it and minimizes.
        Exploration decays toward zero over the allotted search time.
        """
        if self.parent is None:
            return 0.0

        decay_multiplier = 1.0 - (
            min(helperfuncs.decay, 1.0)
            * min(max(time_fraction, 0.0), 1.0)
        )

        explore = helperfuncs.factor * decay_multiplier * np.sqrt(
            np.log(self.parent.visits + 1.0) / (self.visits + 1.0)
        )

        tactical = (
            self.tactical_bonus()
            * helperfuncs.factor
            * decay_multiplier
        )

        return explore + tactical

    def search_score(self, time_fraction):
        """Perspective-aware score used while traversing the tree.

        Traversal intentionally uses the EMA-smoothed value.  The current
        unsmoothed minimax value is kept separately for backup/final choice.
        """
        if self.value is None:
            # Normally every generated child is immediately evaluated.  Keep a
            # safe fallback that makes an uninitialised node attractive.
            return (
                float("inf")
                if self.parent.board.turn == chess.WHITE
                else float("-inf")
            )

        amount = self.exploration_amount(time_fraction)

        # self.parent.board.turn is the player choosing this child.
        if self.parent.board.turn == chess.WHITE:
            return self.value + amount
        else:
            return self.value - amount

    # Keep the old method name for compatibility with external code.
    def ucb(self, time_fraction):
        return self.search_score(time_fraction)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate_nn(self):
        """Evaluate this position with the two-plane ONNX value network."""
        # Plane 0 = board representation.
        # Plane 1 = 1.0 for White to move, 0.0 for Black to move.
        boardlist = fast_board_to_boardmap(self.board)
        whosemove = float(fast_board_to_feature(self.board)[0])

        board_plane = np.asarray(
            boardlist,
            dtype=np.float32,
        ).reshape(1, 1, 8, 8)

        move_plane = np.full(
            (1, 1, 8, 8),
            whosemove,
            dtype=np.float32,
        )

        pos = np.concatenate([board_plane, move_plane], axis=1)
        output = self.net.run(None, {"input": pos})[0]
        return float(np.asarray(output).reshape(-1)[0])

    def evaluate_position(self):
        """Return an exact WHITE-perspective value when one is available.

        claim_draw=False is intentional.  A claimable threefold/50-move draw
        is optional, not an automatically terminal position; that option is
        incorporated into minimax backup in best_child_minimax_value().

        Automatic game endings (mate, stalemate, insufficient material,
        fivefold repetition, 75-move rule, etc.) are terminal here.
        """
        outcome = self.board.result(claim_draw=False)

        if outcome != "*":
            mate_bonus = max((10 - self.depth) / 10.0, 0.0)

            if outcome == "1-0":
                return 2.0 + mate_bonus
            elif outcome == "0-1":
                return -1.0 - mate_bonus
            else:
                return 0.5

        if helperfuncs.TABLEBASE and lt5(self.board):
            result = helperfuncs.TABLEBASE.probe_dtz(self.board)

            # probe_dtz() is relative to the SIDE TO MOVE.  Convert to the
            # fixed White perspective used everywhere else in the engine.
            if 1 <= result <= 100:
                side_to_move_value = 1.0 + (100 - result) / 100.0
            elif -100 <= result <= -1:
                side_to_move_value = 0.0 - (100 + result) / 100.0
            elif result == 0 or result < -100 or result > 100:
                side_to_move_value = 0.5
            else:
                return None

            if self.board.turn == chess.WHITE:
                return side_to_move_value
            else:
                return 1.0 - side_to_move_value

        return None

    # ------------------------------------------------------------------
    # Expansion
    # ------------------------------------------------------------------
    def generate_children(self):
        if self.terminal or self.children:
            return

        all_positions = []
        evaled = []
        not_evaled = []

        legal_moves = list(self.board.legal_moves)
        helperfuncs.nodes += len(legal_moves)

        for move in legal_moves:
            newboard = self.board.copy()
            newboard.push(move)

            newnode = Node(
                newboard,
                self.net,
                move,
                self,
                depth=self.depth + 1,
            )

            # Let evaluate_position() handle all genuinely automatic endings,
            # including the 75-move rule.  Do not treat the claimable 50-move
            # rule as automatically terminal.
            score = newnode.evaluate_position()

            if score is not None:
                exact_value = float(score)
                newnode.value = exact_value
                newnode.minimax_value = exact_value
                newnode.terminal = True
                evaled.append(newnode)
                continue

            # Plane 0: board representation.
            board_plane = np.asarray(
                fast_board_to_boardmap(newboard),
                dtype=np.float32,
            ).reshape(1, 8, 8)

            # Plane 1: side to move.
            whosemove = float(fast_board_to_feature(newboard)[0])
            move_plane = np.full(
                (1, 8, 8),
                whosemove,
                dtype=np.float32,
            )

            # Each child position has shape (2, 8, 8).
            position = np.concatenate(
                [board_plane, move_plane],
                axis=0,
            )

            all_positions.append(position)
            not_evaled.append(newnode)

        if not_evaled:
            # Shape: (batch_size, 2, 8, 8).
            pos = np.asarray(all_positions, dtype=np.float32)
            ort_inputs = {"input": pos}

            while True:
                try:
                    result = self.net.run(None, ort_inputs)[0]
                    break
                except Exception as exc:
                    # Preserve the existing ONNX Runtime recovery behaviour.
                    if helperfuncs.log:
                        print(
                            "info string ONNX inference failed, "
                            f"recreating session: {exc}"
                        )

                    sess_options = onnxruntime.SessionOptions()
                    sess_options.intra_op_num_threads = helperfuncs.num_cores

                    self.net = onnxruntime.InferenceSession(
                        helperfuncs.model_path,
                        sess_options,
                        providers=[helperfuncs.provider],
                    )

                    helperfuncs.broken = True

            result = np.asarray(result).reshape(-1)

            for i, node in enumerate(not_evaled):
                value = float(result[i])

                if node.board.fullmove_number <= helperfuncs.temp_moves:
                    value *= (
                        1.0
                        + random.random()
                        * helperfuncs.temperature
                        / 100.0
                    )

                # Ordinary network values stay inside [0, 1].
                value = min(max(value, 0.0), 1.0)

                # A freshly generated node starts with its NN evaluation as both
                # the traversal value and the current minimax estimate.
                node.value = value
                node.minimax_value = value
                evaled.append(node)

        self.children = evaled

    # ------------------------------------------------------------------
    # Backup helpers
    # ------------------------------------------------------------------
    def best_child_minimax_value(self):
        """
        Propagate fresh information from the best *searched* principal child.
    
        child.value:
            smoothed value used to decide which searched child is currently best
    
        child.minimax_value:
            fresh value propagated upward
    
        Raw generated NN children are deliberately not allowed to become the
        propagated principal variation until they have actually been searched.
        """
    
        if not self.children:
            return self.minimax_value
    
        candidates = [
            child
            for child in self.children
            if child.value is not None
            and child.minimax_value is not None
            and (
                child.terminal
                or child.visits >= 2
            )
        ]
    
        if candidates:
            if self.board.turn == chess.WHITE:
                best_child = max(
                    candidates,
                    key=lambda child: child.value
                )
            else:
                best_child = min(
                    candidates,
                    key=lambda child: child.value
                )
    
            value = best_child.minimax_value
    
        else:
            # We expanded this node, but none of its children has actually
            # been searched beyond its initial NN evaluation yet.
            #
            # Keep the node's existing estimate rather than taking max/min
            # over a batch of noisy raw NN outputs.
            value = self.minimax_value
    
            if value is None:
                value = self.value
    
        # Claimable draw is an optional move worth exactly 0.5.
        if (
            self.board.can_claim_threefold_repetition()
            or self.board.can_claim_fifty_moves()
        ):
            if self.board.turn == chess.WHITE:
                value = max(value, 0.5)
            else:
                value = min(value, 0.5)
    
        return float(value)

    # Retain the old helper name in case other project code calls it.  Its
    # semantics are now deliberately the current unsmoothed minimax value.
    def best_child_value(self):
        return self.best_child_minimax_value()

    def update_value(self, new_minimax_value):
        """Update fresh minimax state and locally EMA-smooth traversal value."""
        new_minimax_value = float(new_minimax_value)

        # Fresh information: replace immediately.
        self.minimax_value = new_minimax_value

        # Traversal information: damp locally to reduce noisy NN swings.
        if self.value is None:
            self.value = new_minimax_value
        else:
            self.value = (
                0.75 * float(self.value)
                + 0.25 * new_minimax_value
            )

    # ------------------------------------------------------------------
    # Main search
    # ------------------------------------------------------------------
    def pns(self, start_time, time_for_this_move) -> "Node":
        if helperfuncs.log:
            print(
                f"info string explore_factor {helperfuncs.factor} "
                f"capture_bonus {helperfuncs.quiescent} "
                f"check_bonus {helperfuncs.check} "
                f"explore_decay {helperfuncs.decay}"
            )

        # Give the root both an initial smoothed/traversal value and an initial
        # current minimax value.
        if self.value is None or self.minimax_value is None:
            exact = self.evaluate_position()
            initial_value = (
                float(exact)
                if exact is not None
                else float(self.evaluate_nn())
            )
            self.value = initial_value
            self.minimax_value = initial_value

        while time.time() - start_time < time_for_this_move:
            # 1. Traverse using the EMA-smoothed value plus exploration.
            target_node = self

            while target_node.children:
                target_node.visits += 1
                time_fraction = (
                    (time.time() - start_time) / time_for_this_move
                    if time_for_this_move > 0
                    else 1.0
                )

                if target_node.board.turn == chess.WHITE:
                    target_node = max(
                        target_node.children,
                        key=lambda child: child.search_score(time_fraction),
                    )
                else:
                    target_node = min(
                        target_node.children,
                        key=lambda child: child.search_score(time_fraction),
                    )

            # 2. Evaluate terminal state or expand the selected leaf.
            target_node.visits += 1

            exact = target_node.evaluate_position()
            if exact is not None:
                leaf_value = float(exact)
                target_node.terminal = True
                target_node.value = leaf_value
                target_node.minimax_value = leaf_value
            else:
                if target_node.value is None:
                    leaf_value = float(target_node.evaluate_nn())
                    leaf_value = min(max(leaf_value, 0.0), 1.0)
                    target_node.value = leaf_value
                    target_node.minimax_value = leaf_value
                else:
                    leaf_value = float(target_node.value)
                    if target_node.minimax_value is None:
                        target_node.minimax_value = leaf_value

                target_node.generate_children()

            # 3. Backpropagate the CURRENT minimax estimate.  Because this loop
            # runs bottom-up and parents read child.minimax_value, a newly found
            # deep correction can reach the root in this single backup pass.
            node = target_node
            while node is not None:
                if node.children:
                    backed_value = node.best_child_minimax_value()
                else:
                    backed_value = node.minimax_value

                if backed_value is not None and not node.terminal:
                    node.update_value(backed_value)

                node = node.parent

        # 4. Select move.
        if not self.children:
            self.generate_children()

        if not self.children:
            raise RuntimeError("Search root has no legal moves")

        max_visits = max(child.visits for child in self.children)

        # Avoid choosing a completely unexplored NN outlier when the search has
        # actually investigated other moves.  If the search had no time to visit
        # any child, all generated children remain eligible.
        min_visits = (
            max(1, int(max_visits * 0.25))
            if max_visits > 0
            else 0
        )

        options = [
            child
            for child in self.children
            if child.visits >= min_visits
            and child.minimax_value is not None
        ]

        # This should not normally be needed, but keeps move selection robust if
        # a future code path ever creates a child without minimax_value.
        if not options:
            options = [
                child
                for child in self.children
                if child.minimax_value is not None
            ]

        if not options:
            raise RuntimeError("Search root children have no minimax values")

        # Exact forced results are reliable and may be outside the NN [0,1]
        # interval, so allow them to override the visit threshold.
        exact_candidates = [
            child
            for child in self.children
            if child.minimax_value is not None
        ]

        if self.board.turn == chess.WHITE:
            forced = [
                child
                for child in exact_candidates
                if child.minimax_value >= 1.0
            ]
            if forced:
                selected = max(
                    forced,
                    key=lambda child: child.minimax_value,
                )
            else:
                selected = max(
                    options,
                    key=lambda child: child.minimax_value,
                )
        else:
            forced = [
                child
                for child in exact_candidates
                if child.minimax_value <= 0.0
            ]
            if forced:
                selected = min(
                    forced,
                    key=lambda child: child.minimax_value,
                )
            else:
                selected = min(
                    options,
                    key=lambda child: child.minimax_value,
                )

        if helperfuncs.log:
            print(
                f"info string root_visits {self.visits} "
                f"max_visits {max_visits} "
                f"best_visits {selected.visits}"
            )

        return selected
