#!/usr/bin/env python3
"""Compare BFS and A* with Manhattan distance on random solvable 8-Puzzle boards.

Usage:
    python3 validate_astar.py SEED RUNS

For example, seed 58 and 10 runs produce
``astar_validation_runs-10_seed-58.csv`` in the current working directory.
Existing files are never overwritten, so experiments with different seeds or
run counts can run concurrently in the same directory.

CSV columns:
    board,bfs_moves,astar_moves,bfs_expanded,astar_expanded

A board is nine digits in row-major order, with 0 representing the blank.
For example, 123456780 is the goal. Read this column as text to retain any
leading zero. Repeated boards are discarded before either search runs, so each
CSV row represents a different solvable board. At most 181,440 runs are possible:
half of the 9! arrangements are solvable for the standard goal.
Using the same seed reproduces the sequence: a shorter experiment is a prefix
of a longer one. Use different seeds for separate random samples.

The search methods follow Lecture 17. BFS returns its expansion count instead
of printing it; both methods return (solution_states, expanded). Each solution
includes both endpoints. Expansion counts exclude the selected goal; A* also
excludes stale heap entries and counts any reopened nodes it expands again.

Each completed row is written and flushed, leaving usable partial results if the
program is interrupted. A missing solution has an empty move-count field.
Disagreements and missing solutions are reported on standard error. The program
finishes all requested runs, then exits with status 1 if any validation failed
(0 otherwise). Agreement is an empirical check, not a proof of optimality.

Only the Python standard library is required.

% python validate_astar.py 4106 25000            
Writing 25000 runs to astar_validation_runs-25000_seed-4106.csv
Completed 25000 runs in 4166.10 seconds; validation failures: 0. Results: astar_validation_runs-25000_seed-4106.csv
"""

import argparse
from collections import deque
import csv
import heapq
from pathlib import Path
import random
import sys
from time import perf_counter


GOAL_STATE = (1, 2, 3,
              4, 5, 6,
              7, 8, 0)
MAX_RUNS = 181_440


def is_solvable(tiles):
    """Test inversion parity for the standard 3x3 goal, excluding the blank."""
    inversions = 0
    for i in range(len(tiles)):
        for j in range(i + 1, len(tiles)):
            if tiles[i] != 0 and tiles[j] != 0 and tiles[i] > tiles[j]:
                inversions += 1
    return inversions % 2 == 0


def generate_solvable_board(rng):
    """Shuffle until a solvable board is found, using the experiment's RNG."""
    tiles = list(range(9))
    rng.shuffle(tiles)
    while not is_solvable(tiles):
        rng.shuffle(tiles)
    return tuple(tiles)


def is_empty(frontier):
    """Check whether the frontier is empty."""
    return len(frontier) == 0


def is_goal(state, goal_state):
    """Determine whether a state matches the goal."""
    return state == goal_state


def expand(state):
    """Generate successors in the lecture's order: left, right, up, down."""
    size = int(len(state) ** 0.5)
    idx = state.index(0)
    x, y = idx % size, idx // size
    neighbors = []

    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dx, dy in moves:
        nx, ny = x + dx, y + dy
        if 0 <= nx < size and 0 <= ny < size:
            n_idx = ny * size + nx
            new_state = list(state)
            new_state[idx], new_state[n_idx] = new_state[n_idx], new_state[idx]
            neighbors.append(tuple(new_state))
    return neighbors


def bfs(initial_state, goal_state):
    """Return (solution_states, expanded) using breadth-first graph search."""
    frontier = deque()
    frontier.append((initial_state, []))  # A node: (state, preceding states).
    reached = {initial_state}
    expanded = 0

    while not is_empty(frontier):
        current_state, path = frontier.popleft()

        if is_goal(current_state, goal_state):
            return path + [current_state], expanded

        expanded += 1
        for neighbor in expand(current_state):
            if neighbor not in reached:
                reached.add(neighbor)
                frontier.append((neighbor, path + [current_state]))

    return None, expanded


def manhattan_distance(state, goal_state):
    """Sum numbered-tile distances to their goal positions; omit the blank."""
    distance = 0
    size = int(len(state) ** 0.5)
    for num in range(1, len(state)):
        idx1 = state.index(num)
        idx2 = goal_state.index(num)
        x1, y1 = idx1 % size, idx1 // size
        x2, y2 = idx2 % size, idx2 // size
        distance += abs(x1 - x2) + abs(y1 - y2)
    return distance


def astar(initial_state, goal_state, heuristic):
    """Return (solution_states, expanded) using the lecture's A* graph search."""
    # Entries contain (f, g, state, preceding states).
    frontier = [(heuristic(initial_state, goal_state), 0, initial_state, [])]
    best_g = {initial_state: 0}
    expanded = 0

    while frontier:
        f, g, current_state, path = heapq.heappop(frontier)
        # A cheaper path may have replaced this queue entry.
        if g > best_g[current_state]:
            continue

        if is_goal(current_state, goal_state):
            return path + [current_state], expanded

        expanded += 1
        for neighbor in expand(current_state):
            new_g = g + 1  # Each puzzle move costs 1.
            if neighbor not in best_g or new_g < best_g[neighbor]:
                best_g[neighbor] = new_g
                new_f = new_g + heuristic(neighbor, goal_state)
                heapq.heappush(
                    frontier, (new_f, new_g, neighbor, path + [current_state])
                )

    return None, expanded


def run_experiments(seed, runs):
    """Analyze unique boards and return (output_path, failure_count)."""
    rng = random.Random(seed)
    output_path = Path(f"astar_validation_runs-{runs}_seed-{seed}.csv")
    failures = 0
    analyzed_boards = {}  # Map each tuple board to its unique run number.

    # Exclusive creation prevents concurrent experiments from overwriting data.
    with output_path.open("x", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow([
            "board", "bfs_moves", "astar_moves", "bfs_expanded", "astar_expanded"
        ])
        output_file.flush()
        print(f"Writing {runs} runs to {output_path}", flush=True)

        while len(analyzed_boards) < runs:
            initial_state = generate_solvable_board(rng)
            if initial_state in analyzed_boards:
                continue

            run = len(analyzed_boards) + 1
            analyzed_boards[initial_state] = run
            board = "".join(str(tile) for tile in initial_state)

            bfs_solution, bfs_expanded = bfs(initial_state, GOAL_STATE)
            astar_solution, astar_expanded = astar(
                initial_state, GOAL_STATE, manhattan_distance
            )
            bfs_moves = len(bfs_solution) - 1 if bfs_solution is not None else None
            astar_moves = (
                len(astar_solution) - 1 if astar_solution is not None else None
            )

            if (bfs_moves is None or astar_moves is None
                    or bfs_moves != astar_moves):
                failures += 1
                print(
                    f"Validation failure on run {run}/{runs}: board={board}, "
                    f"bfs_moves={bfs_moves}, astar_moves={astar_moves}",
                    file=sys.stderr,
                )

            writer.writerow([
                board, bfs_moves, astar_moves, bfs_expanded, astar_expanded
            ])
            output_file.flush()

    return output_path, failures


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare lecture BFS and A* with Manhattan distance on random "
            "unique solvable 8-Puzzle boards. Save a CSV in the current directory."
        ),
        epilog=(
            "Example: %(prog)s 58 10. Repeated boards are skipped before analysis. "
            "The same seed reproduces the same board sequence. Expanded-node "
            "counts exclude selected goals and stale A* entries."
        ),
    )
    parser.add_argument("seed", type=int, help="integer random seed")
    parser.add_argument(
        "runs", type=int,
        help=f"number of unique boards to solve (1 to {MAX_RUNS:,})",
    )
    args = parser.parse_args()
    if not 1 <= args.runs <= MAX_RUNS:
        parser.error(f"runs must be between 1 and {MAX_RUNS:,}")

    started = perf_counter()
    try:
        output_path, failures = run_experiments(args.seed, args.runs)
    except FileExistsError as error:
        parser.error(
            f"output already exists: {error.filename}; choose a different "
            "working directory, seed, or run count"
        )
    except KeyboardInterrupt:
        print("\nInterrupted; completed rows remain in the CSV.", file=sys.stderr)
        return 130

    elapsed = perf_counter() - started
    print(
        f"Completed {args.runs} runs in {elapsed:.2f} seconds; "
        f"validation failures: {failures}. Results: {output_path}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
