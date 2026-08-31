import json
import math
import os

import chess
import chess.engine
import numpy as np
import onnxruntime as ort

from helperfuncs import *


ONNX_PATH = r"C:\Projects\Parakeet\best_parakeet_2.onnx"
STOCKFISH_PATH = r"engines/stockfish-windows/stockfish-windows-x86-64-avx2.exe"
REGRESSION_HISTORY_PATH = ".tester_last_regression.json"
STOCKFISH_TIME_SECONDS = 1.0
STOCKFISH_MATE_SCORE_CP = 100000


# ------------------------------------------------------------
# ONNX setup
# ------------------------------------------------------------

available_providers = ort.get_available_providers()

if "CUDAExecutionProvider" in available_providers:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    print("Running ONNX model on the GPU")
else:
    providers = ["CPUExecutionProvider"]
    print("Running ONNX model on the CPU")


session = ort.InferenceSession(
    ONNX_PATH,
    providers=providers,
)

print("Session providers:", session.get_providers())
print("ONNX input:", session.get_inputs()[0].name)
print("ONNX output:", session.get_outputs()[0].name)


# ------------------------------------------------------------
# Shared position encoding / ONNX helpers
# ------------------------------------------------------------


def board_to_network_input(board):
    """Encode one board exactly as the current two-plane value net expects."""
    boardlist = fast_board_to_boardmap(board)
    whosemove = float(fast_board_to_feature(board)[0])

    board_plane = np.asarray(
        boardlist,
        dtype=np.float32,
    ).reshape(1, 8, 8)

    move_plane = np.full(
        (1, 8, 8),
        whosemove,
        dtype=np.float32,
    )

    # Shape: (2, 8, 8)
    return np.concatenate(
        [board_plane, move_plane],
        axis=0,
    )



def run_network_batch(network_session, batch):
    """Run a (B, 2, 8, 8) batch through an ONNX value network."""
    input_name = network_session.get_inputs()[0].name
    output_name = network_session.get_outputs()[0].name

    output = network_session.run(
        [output_name],
        {input_name: batch.astype(np.float32)},
    )[0]

    return np.asarray(output).reshape(-1).astype(np.float64)



def run_network_single(network_session, board):
    position = board_to_network_input(board)[None, ...]
    return float(run_network_batch(network_session, position)[0])


# ------------------------------------------------------------
# Normal evaluation mode
# ------------------------------------------------------------


def evaluate_current_position(session, board):
    value = run_network_single(session, board)

    print(
        f"Current board NN evaluation: {value:.6f} "
        f"({'White' if board.turn == chess.WHITE else 'Black'} to move)"
    )

    return value



def evaluate_next_moves(session, board):
    """
    Evaluate the current position and every legal child position.

    All legal moves are evaluated in a single ONNX batch.
    """

    print(f"\nPosition: {board.fen()}")
    print(
        f"Side to move: "
        f"{'White' if board.turn == chess.WHITE else 'Black'}"
    )

    evaluate_current_position(session, board)

    move_data = []
    board_maps = []

    for move in board.legal_moves:
        san = board.san(move)

        child = board.copy(stack=False)
        child.push(move)

        move_data.append((move, san))
        board_maps.append(board_to_network_input(child))

    if not move_data:
        print("\nNo legal moves.")
        return []

    # Shape: (number_of_moves, 2, 8, 8)
    batch = np.asarray(
        board_maps,
        dtype=np.float32,
    )

    values = run_network_batch(session, batch)

    results = []

    for (move, san), value in zip(move_data, values):
        results.append(
            (
                move,
                san,
                float(value),
            )
        )

    # Network is White-perspective.
    results.sort(
        key=lambda x: x[2],
        reverse=(board.turn == chess.WHITE),
    )

    print("\nLegal moves")
    print("-" * 48)

    for move, san, value in results:
        print(
            f"{move.uci():6s}  "
            f"{san:8s}  "
            f"{value:.6f}"
        )

    return results


# ------------------------------------------------------------
# Regression helpers
# ------------------------------------------------------------


def cp_to_win_prob(cp):
    """
    Convert a White-POV Stockfish centipawn score to the same target scale
    used to train Parakeet.

    This is algebraically equivalent to the original exponential formula:
        0.4 * (1 - exp(-cp / 200)) / (1 + exp(-cp / 200)) + 0.5

    tanh() is used because it is numerically stable for very large mate scores.
    """
    return 0.4 * math.tanh(float(cp) / 400.0) + 0.5



def stockfish_score_to_values(white_pov_score):
    """
    Return:
        comparable_value: Parakeet target scale, roughly [0.1, 0.9]
        ordering_cp:       numeric White-POV score used to rank moves
        display:           human-readable Stockfish pawn/mate score
    """
    if white_pov_score.is_mate():
        mate = white_pov_score.mate()
        ordering_cp = white_pov_score.score(
            mate_score=STOCKFISH_MATE_SCORE_CP
        )

        if mate is None:
            display = "mate"
        else:
            display = f"#{mate:+d}"
    else:
        ordering_cp = white_pov_score.score()
        if ordering_cp is None:
            ordering_cp = 0
        display = f"{ordering_cp / 100.0:+.2f}"

    if ordering_cp is None:
        ordering_cp = 0

    comparable_value = cp_to_win_prob(ordering_cp)

    return float(comparable_value), float(ordering_cp), display



def analyse_stockfish(stockfish_engine, board):
    """Analyse one board and return White-POV Stockfish values."""
    info = stockfish_engine.analyse(
        board,
        chess.engine.Limit(time=STOCKFISH_TIME_SECONDS),
    )

    white_pov_score = info["score"].white()
    return stockfish_score_to_values(white_pov_score)



def resolve_network_path(name):
    """Resolve a network filename from the current working directory."""
    name = name.strip().strip('"').strip("'")

    if not name:
        return None

    candidates = [name]

    if not name.lower().endswith(".onnx"):
        candidates.append(name + ".onnx")

    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)

    # Return the most likely path so the error message is useful.
    return os.path.abspath(candidates[-1])



def load_regression_history():
    try:
        with open(REGRESSION_HISTORY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        old_name = data.get("old_network")
        new_name = data.get("new_network")

        if old_name and new_name:
            return old_name, new_name
    except (OSError, ValueError, TypeError):
        pass

    return None, None



def save_regression_history(old_name, new_name):
    try:
        with open(REGRESSION_HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "old_network": old_name,
                    "new_network": new_name,
                },
                f,
                indent=2,
            )
    except OSError as exc:
        print(f"Warning: could not save regression defaults: {exc}")



def prompt_for_networks():
    last_old, last_new = load_regression_history()

    print("\nRegression network selection")
    print("----------------------------")
    print("Network files are resolved from the current working directory.")

    if last_old and last_new:
        print("Press Enter to reuse the previous pair.")
        print(f"Previous old network: {last_old}")
        print(f"Previous new network: {last_new}")
        old_prompt = f"Old network [{last_old}]: "
        new_prompt = f"New network [{last_new}]: "
    else:
        old_prompt = "Old network: "
        new_prompt = "New network: "

    old_name = input(old_prompt).strip()
    if not old_name:
        old_name = last_old

    new_name = input(new_prompt).strip()
    if not new_name:
        new_name = last_new

    if not old_name or not new_name:
        print("Two network filenames are required.")
        return None

    old_path = resolve_network_path(old_name)
    new_path = resolve_network_path(new_name)

    if not os.path.isfile(old_path):
        print(f"Old network not found: {old_path}")
        return None

    if not os.path.isfile(new_path):
        print(f"New network not found: {new_path}")
        return None

    # Store filenames relative to the working directory where possible.
    old_saved = os.path.relpath(old_path, os.getcwd())
    new_saved = os.path.relpath(new_path, os.getcwd())
    save_regression_history(old_saved, new_saved)

    return old_path, new_path, old_saved, new_saved



def regression_metrics(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    errors = a - b

    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    return mae, rmse



def format_stockfish_value(result):
    return f"{result['stockfish']:.6f} ({result['stockfish_raw']})"



def print_summary_move(label, result):
    print(f"\n{label}: {result['uci']} ({result['san']})")
    print(f"  old network : {result['old']:.6f}")
    print(f"  new network : {result['new']:.6f}")
    print(f"  Stockfish   : {format_stockfish_value(result)}")



def run_regression(board):
    """Compare two ONNX value nets and Stockfish on the active board."""
    if board.is_game_over(claim_draw=False):
        print("\nThe current board has no legal moves to regress.")
        return

    selected = prompt_for_networks()
    if selected is None:
        return

    old_path, new_path, old_name, new_name = selected

    print("\nLoading regression networks...")
    print(f"Old: {old_name}")
    print(f"New: {new_name}")

    try:
        old_session = ort.InferenceSession(
            old_path,
            providers=providers,
        )
        new_session = ort.InferenceSession(
            new_path,
            providers=providers,
        )
    except Exception as exc:
        print(f"Could not load regression network: {exc}")
        return

    move_data = []
    board_maps = []
    child_boards = []

    for move in board.legal_moves:
        san = board.san(move)
        child = board.copy(stack=False)
        child.push(move)

        move_data.append((move, san))
        child_boards.append(child)
        board_maps.append(board_to_network_input(child))

    if not move_data:
        print("\nNo legal moves.")
        return

    batch = np.asarray(board_maps, dtype=np.float32)

    try:
        old_values = run_network_batch(old_session, batch)
        new_values = run_network_batch(new_session, batch)
        old_root = run_network_single(old_session, board)
        new_root = run_network_single(new_session, board)
    except Exception as exc:
        print(f"ONNX regression inference failed: {exc}")
        return

    stockfish_engine = None

    try:
        print(f"Starting Stockfish: {STOCKFISH_PATH}")
        stockfish_engine = chess.engine.SimpleEngine.popen_uci(
            STOCKFISH_PATH
        )

        print(
            f"Analysing root with Stockfish "
            f"({STOCKFISH_TIME_SECONDS:.1f}s)..."
        )
        sf_root, sf_root_order, sf_root_raw = analyse_stockfish(
            stockfish_engine,
            board,
        )

        stockfish_values = []

        for i, child in enumerate(child_boards, start=1):
            print(
                f"\rStockfish analysing legal move "
                f"{i}/{len(child_boards)}...",
                end="",
                flush=True,
            )

            sf_value, sf_order, sf_raw = analyse_stockfish(
                stockfish_engine,
                child,
            )

            stockfish_values.append(
                (sf_value, sf_order, sf_raw)
            )

        print()

    except (OSError, chess.engine.EngineError, chess.engine.EngineTerminatedError) as exc:
        print(f"\nStockfish regression failed: {exc}")
        return
    finally:
        if stockfish_engine is not None:
            try:
                stockfish_engine.quit()
            except Exception:
                pass

    results = []

    for (move, san), old_value, new_value, sf_data in zip(
        move_data,
        old_values,
        new_values,
        stockfish_values,
    ):
        sf_value, sf_order, sf_raw = sf_data

        results.append(
            {
                "move": move,
                "uci": move.uci(),
                "san": san,
                "old": float(old_value),
                "new": float(new_value),
                "stockfish": float(sf_value),
                "stockfish_order": float(sf_order),
                "stockfish_raw": sf_raw,
            }
        )

    white_to_move = board.turn == chess.WHITE

    # Sort the table by Stockfish preference for the side to move.
    results.sort(
        key=lambda r: r["stockfish_order"],
        reverse=white_to_move,
    )

    stockfish_best = (
        max(results, key=lambda r: r["stockfish_order"])
        if white_to_move
        else min(results, key=lambda r: r["stockfish_order"])
    )

    old_best = (
        max(results, key=lambda r: r["old"])
        if white_to_move
        else min(results, key=lambda r: r["old"])
    )

    new_best = (
        max(results, key=lambda r: r["new"])
        if white_to_move
        else min(results, key=lambda r: r["new"])
    )

    print("\nRegression")
    print("==========")
    print(f"Position: {board.fen()}")
    print(
        f"Side to move: "
        f"{'White' if white_to_move else 'Black'}"
    )
    print(f"Old network: {old_name}")
    print(f"New network: {new_name}")
    print(
        "Stockfish column is converted to Parakeet's training target scale; "
        "the raw Stockfish pawn/mate score is shown in parentheses."
    )

    print("\nCurrent position")
    print("----------------")
    print(f"old network : {old_root:.6f}")
    print(f"new network : {new_root:.6f}")
    print(f"Stockfish   : {sf_root:.6f} ({sf_root_raw})")

    print("\nLegal moves")
    print("-" * 78)
    print(
        f"{'UCI':6s}  "
        f"{'SAN':10s}  "
        f"{'Old NN':>10s}  "
        f"{'New NN':>10s}  "
        f"{'Stockfish':>23s}"
    )
    print("-" * 78)

    for result in results:
        print(
            f"{result['uci']:6s}  "
            f"{result['san']:10s}  "
            f"{result['old']:10.6f}  "
            f"{result['new']:10.6f}  "
            f"{format_stockfish_value(result):>23s}"
        )

    old_array = [r["old"] for r in results]
    new_array = [r["new"] for r in results]
    sf_array = [r["stockfish"] for r in results]

    old_new_mae, old_new_rmse = regression_metrics(
        old_array,
        new_array,
    )
    old_sf_mae, old_sf_rmse = regression_metrics(
        old_array,
        sf_array,
    )
    new_sf_mae, new_sf_rmse = regression_metrics(
        new_array,
        sf_array,
    )

    print("\nSummary")
    print("=======")

    print_summary_move("Stockfish best move", stockfish_best)
    print_summary_move("Old network best move", old_best)
    print_summary_move("New network best move", new_best)

    print("\nError metrics across all legal child positions")
    print("------------------------------------------------")
    print(f"{'Comparison':24s}  {'MAE':>10s}  {'RMSE':>10s}")
    print(f"{'old vs new':24s}  {old_new_mae:10.6f}  {old_new_rmse:10.6f}")
    print(f"{'old vs Stockfish':24s}  {old_sf_mae:10.6f}  {old_sf_rmse:10.6f}")
    print(f"{'new vs Stockfish':24s}  {new_sf_mae:10.6f}  {new_sf_rmse:10.6f}")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------


def chess_cli(session):
    board = chess.Board()

    print("\nChess evaluation CLI")
    print("--------------------")
    print(
        "Enter moves in UCI format, "
        "e.g. d2d4, g8f6, e7e8q"
    )
    print("Commands:")
    print("  board             - show the current board")
    print("  fen               - show the current FEN")
    print("  fen <FEN string>  - load a position from FEN")
    print("  undo              - undo the previous move")
    print("  reset             - reset to the starting position")
    print("  eval              - evaluate current position's legal moves")
    print("  regression        - compare two networks + Stockfish here")
    print("  quit              - exit")
    print()

    evaluate_next_moves(session, board)

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue

        command = user_input.lower()

        if command in {"quit", "exit", "q"}:
            break

        if command == "board":
            print()
            print(board)
            continue

        # Load FEN
        if command.startswith("fen "):
            # Keep original capitalization of FEN!
            fen_string = user_input[4:].strip()

            try:
                board = chess.Board(fen_string)
            except ValueError as e:
                print(f"Invalid FEN: {e}")
                continue

            print("\nLoaded FEN position:")
            print(board)

            print("Valid position:", board.is_valid())

            if board.is_game_over():
                print(
                    f"\nGame over: "
                    f"{board.result(claim_draw=True)}"
                )
            else:
                evaluate_next_moves(
                    session,
                    board,
                )

            continue

        if command == "fen":
            print(board.fen())
            continue

        if command == "reset":
            board.reset()

            print("\nBoard reset.")
            print(board)

            evaluate_next_moves(
                session,
                board,
            )

            continue

        if command == "undo":
            if board.move_stack:
                undone = board.pop()

                print(
                    f"\nUndid {undone.uci()}"
                )

                print(board)

                evaluate_next_moves(
                    session,
                    board,
                )
            else:
                print("No moves to undo.")

            continue

        if command == "eval":
            evaluate_next_moves(
                session,
                board,
            )
            continue

        if command in {"regression", "regress", "compare"}:
            run_regression(board)
            continue

        # Treat everything else as a UCI move
        try:
            move = chess.Move.from_uci(
                user_input.lower()
            )
        except ValueError:
            print(
                f"Invalid UCI move: "
                f"{user_input}"
            )
            continue

        if move not in board.legal_moves:
            print(
                f"Illegal move: "
                f"{user_input}"
            )
            continue

        san = board.san(move)
        board.push(move)

        print(
            f"\nPlayed: "
            f"{move.uci()} ({san})"
        )
        print(board)

        if board.is_game_over():
            print(
                f"\nGame over: "
                f"{board.result(claim_draw=True)}"
            )
        else:
            evaluate_next_moves(
                session,
                board,
            )


# ------------------------------------------------------------
# Start CLI
# ------------------------------------------------------------


if __name__ == "__main__":
    chess_cli(session)
