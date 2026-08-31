import numpy as np
import torch
import torch.nn as nn
import time
import os
import onnxruntime as ogpu
from helperfuncs import *
from nn_creator import *
from data_loader import *
import gc
import psutil

# configuring device
try:
    device = xm.xla_device()
    print("Running on the TPU")
except:
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('Running on the GPU')
        torch.cuda.synchronize()
    else:
        device = torch.device('cpu')
        print('Running on the CPU')


LONG_INTERVAL = 200
SHORT_INTERVAL = 10

EVAL_SET_SIZE = 65536


# Total (rough, assuming equal training time):
# Easy puzzles: 22.4%, Mid puzzles: 19.3%, Hard puzzles: 12.0%, Mates: 7.8%, Openings: 10.9%, Midgames: 14.1%, Endgames: 13.5%

EPOCH_SIZE = 65536
BATCH_SIZE = 4096
VAL_SIZE = 65536 * 12

learning_rate = 0.00008
l2r = 0
tvl = 0
checkpoint = -1
save = True
model_name = "parakeet_1"

TIME_CHECK = True

process = psutil.Process(os.getpid())

def print_memory(tag=""):
    ram_gb = process.memory_info().rss / (1024 ** 3)

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)

        print(
            f"\nMEM {tag}: "
            f"RAM={ram_gb:.2f} GB | "
            f"VRAM alloc={allocated:.2f} GB | "
            f"reserved={reserved:.2f} GB | "
            f"peak={max_allocated:.2f} GB"
        )
    else:
        print(f"\nMEM {tag}: RAM={ram_gb:.2f} GB")


print("Dataset info:")
print(f"- Drawish positions: 65536 * {NDP} = {65536 * NDP}")
print(f"- Advantageous positions: 65536 * {NAP} = {65536 * NAP}")
print(f"- Winning positions: 65536 * {NWP} = {65536 * NWP}")
print(f"- Total = {65536 * (NAP + NDP + NWP)}")

print("----------------")


# Build validation sets
print("Target validation set size:", VAL_SIZE)

val_draw_boards, val_draw_features, val_draw_evals = [], [], []

for dvs in DRAW_VALIDATION_SET:
    vb, vf, ve = read_eval_data("draw", dvs)
    val_draw_boards += vb
    val_draw_features += vf
    val_draw_evals += ve
print("Draws board validation set dimensions: ", np.shape(val_draw_boards))

val_adv_boards, val_adv_features, val_adv_evals = [], [], []

for avs in ADV_VALIDATION_SET:
    vb, vf, ve = read_eval_data("adv", avs)
    val_adv_boards += vb
    val_adv_features += vf
    val_adv_evals += ve
print("Advs board validation set dimensions: ", np.shape(val_adv_boards))

val_win_boards, val_win_features, val_win_evals = [], [], []

for wvs in WIN_VALIDATION_SET:
    vb, vf, ve = read_eval_data("winning", wvs)
    val_win_boards += vb
    val_win_features += vf
    val_win_evals += ve
print("Win board validation set dimensions: ", np.shape(val_win_boards))


val_position_list = list(val_draw_boards) + list(val_adv_boards) + list(val_win_boards)
val_features_list = list(val_draw_features) + list(val_adv_features) + list(val_win_features)
val_eval_list = list(val_draw_evals) + list(val_adv_evals) + list(val_win_evals)

print("Total board validation set dimensions: ", np.shape(val_position_list))

# Convert validation data to the same 2-plane representation used for training:
#   channel 0 = piece board
#   channel 1 = side to move (1 = White, 0 = Black)
def add_turn_plane(board_list, feature_list):
    boards = np.asarray(board_list, dtype=np.float32).reshape(-1, 1, 8, 8)
    turns = np.asarray([feature[0] for feature in feature_list], dtype=np.float32)
    turn_planes = np.broadcast_to(
        turns[:, None, None, None],
        (len(turns), 1, 8, 8),
    )
    return np.concatenate((boards, turn_planes), axis=1).astype(np.float32, copy=False)

val_position_list = add_turn_plane(val_position_list, val_features_list)
print("2-plane validation set dimensions: ", val_position_list.shape)


model = ChessEvaluationNet().to(device=device)


from pytorch_optimizer import SOAP
optimizer = SOAP(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()


if checkpoint == 0:
    num_epoch = 0
    best_vloss = 1000
    model.train()
    with open(f"D:/parakeet/models/{model_name}_loss.csv", "w") as file:
        file.write("")
    print("Using new models.")

elif checkpoint == -1:
    print("Loading from last checkpoint.")
    checkpoint_file = torch.load(f"D:/parakeet/models/{model_name}.pickle", weights_only=True, map_location=device)
    model.load_state_dict(checkpoint_file["model_state_dict"])
    model.to(device=device)
    optimizer.load_state_dict(checkpoint_file["optimizer_state_dict"])

    # Update learning rate in case it is changed midway.
    for g in optimizer.param_groups:
      g['lr'] = learning_rate

    num_epoch = checkpoint_file["epoch"] + 1
    best_vloss = checkpoint_file["best_loss"]
    model.train()
    print(f"Best validation loss at checkpoint: {best_vloss}")

elif checkpoint == -2:
    print("Switching to new dataset.")
    checkpoint_file = torch.load(f"D:/parakeet/models/{model_name}.pickle", weights_only=True, map_location=device)
    model.load_state_dict(checkpoint_file["model_state_dict"])
    model.to(device=device)

    # Update learning rate in case it is changed midway.
    for g in optimizer.param_groups:
      g['lr'] = learning_rate

    num_epoch = 0
    best_vloss = 1000
    model.train()


else:
    print(f"Loading from epoch {checkpoint}.")
    checkpoint_file = torch.load(f"D:/parakeet/models/{model_name}_{checkpoint}.pickle", weights_only=True, map_location=device)
    model.load_state_dict(checkpoint_file["model_state_dict"])
    model.to(device=device)
    optimizer.load_state_dict(checkpoint_file["optimizer_state_dict"])

    # Update learning rate in case it is changed midway.
    for g in optimizer.param_groups:
      g['lr'] = learning_rate

    num_epoch = checkpoint_file["epoch"] + 1
    best_vloss = checkpoint_file["best_loss"]
    model.train()
    print(f"Best validation loss at checkpoint: {best_vloss}")



print(model)
print("Batch size = ", BATCH_SIZE)
print("Validation size = ", VAL_SIZE)
print("L2 regularisation strength =", l2r)
print("Learning rate =", learning_rate)

def warmup_then_expo(epoch):
  if epoch < 58:
    return epoch / 58
  else:
    return (0.999 ** (epoch - 58))
#scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_then_expo)

tlr, tl, vl = [], [], []

running_mean = 0
M2 = 0
readings = 0

# Setup dataloaders
draw_loader = DataLoader(position.DRAW)
adv_loader = DataLoader(position.ADVANTAGE)
win_loader = DataLoader(position.WINNING)

for epoch in range(num_epoch, 800000):

    bl, el = [], []
    bl3, el3, bl4, el4, bl5, el5 = [], [], [], [], [], []
    start = time.time()

    bl3, el3 = draw_loader.get_data(round(EPOCH_SIZE * 0.4))
    bl4, el4 = adv_loader.get_data(round(EPOCH_SIZE * 0.3))
    bl5, el5 = win_loader.get_data(round(EPOCH_SIZE * 0.3))

    bl = np.concatenate((bl3, bl4, bl5), axis=0)
    el = np.concatenate((el3, el4, el5), axis=0)
    #print("Shape of training set boards:", np.shape(bl))

    bl = np.asarray(bl, dtype=np.float32)
    el = np.asarray(el, dtype=np.float32)

    if bl.shape != (EPOCH_SIZE, 2, 8, 8):
        print("Unexpected training input shape:", bl.shape)
        continue

    perm = np.random.permutation(len(bl))

    bl = bl[perm]
    el = el[perm]

    cl = 0
    clr = 0
    norms = []
    data_load_time = time.time() - start
    readings += 1
    delta = data_load_time - running_mean
    running_mean += delta / readings
    delta2 = data_load_time - running_mean
    M2 += (delta * delta2)
    variance = M2 / readings
    print(f"Data loaded in {data_load_time} seconds. Mean {running_mean}, Stdev {variance ** 0.5}")
    if (len(bl) != EPOCH_SIZE) or (len(el) != EPOCH_SIZE):
        print("Dataset is invalid.")
        continue
    else:
        start = time.time()
        for batch in range(EPOCH_SIZE // BATCH_SIZE):
            tb = torch.tensor(bl[batch * BATCH_SIZE : (batch + 1) * BATCH_SIZE], device=device, dtype=torch.float).reshape(BATCH_SIZE, 2, 8, 8)
            te = torch.tensor(el[batch * BATCH_SIZE : (batch + 1) * BATCH_SIZE], device=device, dtype=torch.float).reshape(BATCH_SIZE, 1)

            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(tb)
                loss = loss_fn(out, te)
                l = loss.item()
                cl += l
                if l2r != 0:
                    l2 = sum(p.pow(2).sum() for p in model.parameters())
                    loss += l2 * l2r
                    llr = loss.item()
                    clr += llr
            loss.backward()


            optimizer.step()
            tl.append(l)
            if l2r != 0:
                tlr.append(llr)
            completion = int(20 * batch / (EPOCH_SIZE // BATCH_SIZE)) + 1
            if l2r != 0:
                print("\r" + f"[{'-' * completion} {' ' * (20 - completion)}]     Loss {round(l, 6)}, Regularised loss {round(llr, 6)}", end = "")
            else:
                print("\r" + f"[{'-' * completion} {' ' * (20 - completion)}]     Loss {round(l, 6)}", end = "")
        if l2r != 0: print(f"\nEpoch {epoch}, loss {round(cl / (EPOCH_SIZE // BATCH_SIZE), 6)}, regularised loss {round(clr / (EPOCH_SIZE // BATCH_SIZE), 6)}, completed in {time.time() - start} seconds.")
        else: print(f"\nEpoch {epoch}, loss {round(cl / (EPOCH_SIZE // BATCH_SIZE), 6)}, completed in {time.time() - start} seconds.")
        del tb, te
        gc.collect()

        if (epoch % SHORT_INTERVAL == 0):
            with torch.inference_mode():
                model.eval()
                tvl = 0
                start = time.time()
                for vbatch in range(VAL_SIZE // BATCH_SIZE):
                    vp = torch.tensor(val_position_list[vbatch * BATCH_SIZE : (vbatch + 1) * BATCH_SIZE], device=device, dtype=torch.float).reshape(BATCH_SIZE, 2, 8, 8)
                    ve = torch.tensor(val_eval_list[vbatch * BATCH_SIZE : (vbatch + 1) * BATCH_SIZE], device=device, dtype=torch.float).reshape(BATCH_SIZE, 1)
                    out = model(vp)
                    loss = loss_fn(out, ve)
                    tvl += loss.item()

                print("Validation loss", round(tvl / (VAL_SIZE // BATCH_SIZE), 6), "completed in", time.time() - start, "seconds.")
                vl.append(tvl / (VAL_SIZE // BATCH_SIZE))
                if (tvl / (VAL_SIZE // BATCH_SIZE)) < best_vloss:
                    print("New best model!")
                    best_vloss = tvl / (VAL_SIZE // BATCH_SIZE)
                    torch.save(model.state_dict(), f"D:/parakeet/models/best_{model_name}.pickle")

                model.train()
                del vp, ve, out, loss
                gc.collect()

        #before_lr = learning_rate
        #learning_rate *= 2
        #for g in optimizer.param_groups:
        #    g['lr'] = learning_rate
        #print(f"Epoch {epoch} : lr {before_lr} -> {learning_rate}")


        if save:
            if epoch % LONG_INTERVAL == 0:
                torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "best_loss": best_vloss}, f"D:/parakeet/models/{model_name}_{epoch}.pickle")
            if epoch % SHORT_INTERVAL == 0:
                torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "best_loss": best_vloss}, f"D:/parakeet/models/{model_name}.pickle")
            try:
                os.remove(f"D:/parakeet/models/{model_name}_{epoch - LONG_INTERVAL}.pickle")
            except:
                pass

        with open(f"D:/parakeet/models/{model_name}_loss.csv", "a") as file:
            file.write(f"{epoch}, {round(cl / (EPOCH_SIZE // BATCH_SIZE), 6)}, {round(tvl / (VAL_SIZE // BATCH_SIZE), 6)}\n")
        print_memory(f"epoch {epoch}")