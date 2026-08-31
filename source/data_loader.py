from helperfuncs import *
import pickle
import random

def read_eval_data(phase, num):
    data_list = pickle.load(open(f"D:/parakeet/evaluation_database/{phase}_data_{num}.chess", "rb"))
    return data_list

NDP = 528
NAP = 462
NWP = 396


# Training data loader

class DataLoader:

    def __init__(self, dataset):
        self.dataset = dataset
        self.pointer = 0
        self.length = 65536
        self.data = None

    def get_dataset(self):

        done=False

        if self.dataset == position.DRAW:
            ri = random.randint(0, NDP)
            while ri in DRAW_VALIDATION_SET:
                ri = random.randint(0, NDP)
            while not done:
                try:
                    self.data = read_eval_data("draw", ri)
                    done = True
                except:
                    print("Could not read draw", ri)
                ri = random.randint(0, NDP)
        elif self.dataset == position.ADVANTAGE:
            ri = random.randint(0, NAP)
            while ri in ADV_VALIDATION_SET:
                ri = random.randint(0, NAP)
            while not done:
                try:
                    self.data = read_eval_data("adv", ri)
                    done = True
                except:
                    print("Could not read adv", ri)
                ri = random.randint(0, NAP)
        elif self.dataset == position.WINNING:
            ri = random.randint(0, NWP)
            while ri in WIN_VALIDATION_SET:
                ri = random.randint(0, NWP)
            while not done:
                try:
                    self.data = read_eval_data("winning", ri)
                    done = True
                except:
                    print("Could not read winning", ri)
                ri = random.randint(0, NWP)

    def get_data(self, num):
        """Return (positions, evaluations).

        positions has shape (N, 2, 8, 8):
            channel 0 = stored piece board
            channel 1 = side-to-move plane (1 White, 0 Black)
        """
        pos_chunks = []
        eval_chunks = []

        if self.data is None:
            self.get_dataset()

        remaining = num
        while remaining > 0:
            available = self.length - self.pointer
            take = min(remaining, available)
            start = self.pointer
            end = start + take

            # Stored boards are (take, 1, 8, 8).
            boards = np.asarray(self.data[0][start:end], dtype=np.float32).reshape(
                take, 1, 8, 8
            )

            # feature[0] is int(board.turn): 1 = White, 0 = Black.
            turns = np.asarray(
                [feature[0] for feature in self.data[1][start:end]],
                dtype=np.float32,
            )
            turn_planes = np.broadcast_to(
                turns[:, None, None, None],
                (take, 1, 8, 8),
            )

            pos_chunks.append(
                np.concatenate((boards, turn_planes), axis=1).astype(
                    np.float32, copy=False
                )
            )
            eval_chunks.append(
                np.asarray(self.data[2][start:end], dtype=np.float32)
            )

            self.pointer = end
            remaining -= take

            if self.pointer >= self.length:
                self.pointer = 0
                self.get_dataset()

        return np.concatenate(pos_chunks, axis=0), np.concatenate(eval_chunks, axis=0)
