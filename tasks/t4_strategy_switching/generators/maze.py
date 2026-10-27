"""Generate maze instances where wall-following fails but BFS succeeds.

Each maze has:
- An outer boundary wall (connected to the border)
- An interior "island" wall component (disconnected from outer boundary)
- Start (S) on the outer boundary corridor
- Exit (E) on the interior island
- Wall-following from S traces the outer boundary and returns to S
- BFS from S to E finds a valid path through open space
"""

import json
import random
from collections import deque
from pathlib import Path


# ---------------------------------------------------------------------------
# Maze generation
# ---------------------------------------------------------------------------

def make_empty_grid(h, w):
    """Create a grid filled with open cells, bordered by walls."""
    grid = [['.' for _ in range(w)] for _ in range(h)]
    for r in range(h):
        grid[r][0] = '#'
        grid[r][w - 1] = '#'
    for c in range(w):
        grid[0][c] = '#'
        grid[h - 1][c] = '#'
    return grid


def add_interior_walls_prim(grid, h, w, rng, wall_density=0.35):
    """Add maze-like interior walls using a randomised approach.

    We place walls on even-row, even-col intersections and carve passages,
    producing a maze-like structure attached to the outer boundary.
    """
    # Mark candidate wall positions (even rows/cols inside the border)
    candidates = []
    for r in range(2, h - 2, 2):
        for c in range(2, w - 2, 2):
            candidates.append((r, c))

    rng.shuffle(candidates)
    placed = 0
    target = int(len(candidates) * wall_density)

    for (r, c) in candidates:
        if placed >= target:
            break
        # Place a wall segment from this point toward a random border
        direction = rng.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
        dr, dc = direction
        cr, cc = r, c
        segment = []
        while 0 < cr < h - 1 and 0 < cc < w - 1:
            segment.append((cr, cc))
            cr += dr
            cc += dc
        # Only place if it connects to the outer wall
        if len(segment) > 1:
            for sr, sc in segment:
                grid[sr][sc] = '#'
            placed += 1


def get_connected_component(grid, h, w, start_r, start_c, char='#'):
    """BFS to find all cells of `char` connected to (start_r, start_c)."""
    visited = set()
    q = deque([(start_r, start_c)])
    while q:
        r, c = q.popleft()
        if (r, c) in visited:
            continue
        if r < 0 or r >= h or c < 0 or c >= w:
            continue
        if grid[r][c] != char:
            continue
        visited.add((r, c))
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            q.append((r + dr, c + dc))
    return visited


def place_island(grid, h, w, rng):
    """Place a rectangular interior island wall that is NOT connected to the outer boundary.

    Returns the set of island wall cells, or None if placement failed.
    """
    # Try to place island in the interior
    for _ in range(200):
        iw = rng.randint(3, max(3, w // 3))
        ih = rng.randint(3, max(3, h // 3))
        # Must be well inside the grid (at least 2 cells from border)
        top = rng.randint(3, h - ih - 3)
        left = rng.randint(3, w - iw - 3)

        # Check that the island area and a 1-cell buffer are all open
        can_place = True
        for r in range(top - 1, top + ih + 1):
            for c in range(left - 1, left + iw + 1):
                if r <= 0 or r >= h - 1 or c <= 0 or c >= w - 1:
                    can_place = False
                    break
                if grid[r][c] == '#':
                    can_place = False
                    break
            if not can_place:
                break

        if not can_place:
            continue

        # Place the island walls (rectangle outline)
        island_cells = set()
        for r in range(top, top + ih):
            for c in range(left, left + iw):
                if r == top or r == top + ih - 1 or c == left or c == left + iw - 1:
                    grid[r][c] = '#'
                    island_cells.add((r, c))
                else:
                    # Interior of island is open (could be a room)
                    grid[r][c] = '.'

        # Verify island is NOT connected to outer boundary
        outer_walls = get_connected_component(grid, h, w, 0, 0, '#')
        if island_cells & outer_walls:
            # Undo
            for r, c in island_cells:
                grid[r][c] = '.'
            continue

        return island_cells, top, left, ih, iw

    return None


def add_dead_ends_and_corridors(grid, h, w, rng, count=8):
    """Add short dead-end spurs off existing walls to make the maze look more complex."""
    outer_walls = get_connected_component(grid, h, w, 0, 0, '#')

    for _ in range(count * 10):
        if count <= 0:
            break
        # Pick a random open cell adjacent to an outer wall
        r = rng.randint(2, h - 3)
        c = rng.randint(2, w - 3)
        if grid[r][c] != '.':
            continue

        # Check adjacency to outer wall
        adj_wall = False
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if (r + dr, c + dc) in outer_walls:
                adj_wall = True
                break

        if not adj_wall:
            continue

        # Try to extend a spur in a direction away from the wall
        direction = rng.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
        dr, dc = direction
        spur_len = rng.randint(1, 3)
        spur = []
        cr, cc = r, c
        valid = True
        for _ in range(spur_len):
            if 1 < cr < h - 2 and 1 < cc < w - 2 and grid[cr][cc] == '.':
                spur.append((cr, cc))
                cr += dr
                cc += dc
            else:
                valid = False
                break

        if valid and spur:
            for sr, sc in spur:
                grid[sr][sc] = '#'
            # Re-check open connectivity
            open_cells = [(rr, cc2) for rr in range(1, h - 1)
                          for cc2 in range(1, w - 1) if grid[rr][cc2] == '.']
            if open_cells:
                reachable = set()
                q = deque([open_cells[0]])
                while q:
                    cr2, cc2 = q.popleft()
                    if (cr2, cc2) in reachable:
                        continue
                    reachable.add((cr2, cc2))
                    for ddr, ddc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr2 + ddr, cc2 + ddc
                        if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] == '.' and (nr, nc) not in reachable:
                            q.append((nr, nc))
                if len(reachable) < len(open_cells) * 0.8:
                    # Undo - broke connectivity too much
                    for sr, sc in spur:
                        grid[sr][sc] = '.'
                    continue
            count -= 1


def ensure_open_connectivity(grid, h, w):
    """Make sure all open cells are connected. Carve paths if needed."""
    open_cells = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if grid[r][c] in ('.', 'S', 'E'):
                open_cells.append((r, c))

    if not open_cells:
        return False

    visited = set()
    q = deque([open_cells[0]])
    while q:
        r, c = q.popleft()
        if (r, c) in visited:
            continue
        visited.add((r, c))
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] in ('.', 'S', 'E') and (nr, nc) not in visited:
                q.append((nr, nc))

    open_set = set(open_cells)
    if visited == open_set:
        return True

    # Carve to connect disconnected components
    disconnected = open_set - visited
    for cell in disconnected:
        # Find nearest connected cell and carve
        best = None
        best_d = float('inf')
        for vc in visited:
            d = abs(cell[0] - vc[0]) + abs(cell[1] - vc[1])
            if d < best_d:
                best_d = d
                best = vc
        r, c = cell
        tr, tc = best
        while r != tr:
            if grid[r][c] == '#' and 0 < r < h - 1:
                grid[r][c] = '.'
            r += 1 if tr > r else -1
        while c != tc:
            if grid[r][c] == '#' and 0 < c < w - 1:
                grid[r][c] = '.'
            c += 1 if tc > c else -1
        visited.add(cell)

    return True


# ---------------------------------------------------------------------------
# Wall-following (right-hand rule)
# ---------------------------------------------------------------------------

# Directions: 0=up, 1=right, 2=down, 3=left
DR = [-1, 0, 1, 0]
DC = [0, 1, 0, -1]
DIR_NAMES = ['up', 'right', 'down', 'left']


def wall_follow(grid, h, w, start, max_steps=500, extra_steps_after_loop=4):
    """Right-hand rule wall-following from start.

    Returns (trace, returned_to_start, loop_index).
    trace includes extra_steps_after_loop steps past the loop closure.
    The agent faces 'right' initially, with the wall on its right.
    """
    sr, sc = start
    # Find initial facing: face along the corridor (away from wall on right)
    # We need to find a direction where: the cell ahead is open, and the cell
    # to the right is a wall (so we can follow with right hand on wall)

    facing = None
    for f in range(4):
        # Cell ahead
        ar, ac = sr + DR[f], sc + DC[f]
        # Cell to the right
        right_f = (f + 1) % 4
        rr, rc = sr + DR[right_f], sc + DC[right_f]
        if (0 <= ar < h and 0 <= ac < w and grid[ar][ac] in ('.', 'E', 'S') and
                0 <= rr < h and 0 <= rc < w and grid[rr][rc] == '#'):
            facing = f
            break

    if facing is None:
        # Try any direction where ahead is open
        for f in range(4):
            ar, ac = sr + DR[f], sc + DC[f]
            if 0 <= ar < h and 0 <= ac < w and grid[ar][ac] in ('.', 'E', 'S'):
                facing = f
                break

    if facing is None:
        return [], False

    trace = [(sr, sc)]
    r, c = sr, sc
    facings = [facing]

    for step in range(max_steps):
        # Right-hand rule: try to turn right, then go straight, then left, then back
        turned = False
        for turn in [1, 0, -1, 2]:  # right, straight, left, reverse
            new_facing = (facing + turn) % 4
            nr, nc = r + DR[new_facing], c + DC[new_facing]
            if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] in ('.', 'S', 'E'):
                facing = new_facing
                r, c = nr, nc
                turned = True
                break

        if not turned:
            break

        trace.append((r, c))
        facings.append(facing)

        # Check if returned to start (after at least a few steps)
        if (r, c) == (sr, sc) and len(trace) > 4:
            # Continue for a few more steps past the loop closure
            loop_idx = len(trace) - 1
            for extra in range(extra_steps_after_loop):
                turned2 = False
                for turn in [1, 0, -1, 2]:
                    new_facing2 = (facing + turn) % 4
                    nr2, nc2 = r + DR[new_facing2], c + DC[new_facing2]
                    if 0 <= nr2 < h and 0 <= nc2 < w and grid[nr2][nc2] in ('.', 'S', 'E'):
                        facing = new_facing2
                        r, c = nr2, nc2
                        turned2 = True
                        break
                if not turned2:
                    break
                trace.append((r, c))
                facings.append(facing)
            return trace, True, loop_idx

    return trace, False, -1


# ---------------------------------------------------------------------------
# BFS
# ---------------------------------------------------------------------------

def bfs(grid, h, w, start, end):
    """BFS from start to end. Returns path as list of (r, c) or None."""
    sr, sc = start
    er, ec = end
    visited = set()
    q = deque([(sr, sc, [(sr, sc)])])
    while q:
        r, c, path = q.popleft()
        if (r, c) == (er, ec):
            return path
        if (r, c) in visited:
            continue
        visited.add((r, c))
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] != '#' and (nr, nc) not in visited:
                q.append((nr, nc, path + [(nr, nc)]))
    return None


# ---------------------------------------------------------------------------
# Maze generation orchestrator
# ---------------------------------------------------------------------------

def generate_maze(rng, h=None, w=None):
    """Generate a single maze instance.

    Returns dict with maze data, or None on failure.
    """
    if h is None:
        h = rng.randint(12, 16)
    if w is None:
        w = rng.randint(14, 20)

    # Ensure even dimensions for cleaner structure
    if h % 2 == 0:
        h += 1
    if w % 2 == 0:
        w += 1

    grid = make_empty_grid(h, w)

    # Add some interior walls connected to boundary
    add_interior_walls_prim(grid, h, w, rng, wall_density=rng.uniform(0.2, 0.4))

    # Ensure open space is connected before placing island
    ensure_open_connectivity(grid, h, w)

    # Place interior island
    result = place_island(grid, h, w, rng)
    if result is None:
        return None
    island_cells, itop, ileft, ih, iw = result

    # Add dead ends for complexity
    add_dead_ends_and_corridors(grid, h, w, rng, count=rng.randint(3, 8))

    # Place Start on outer corridor (adjacent to outer wall)
    outer_walls = get_connected_component(grid, h, w, 0, 0, '#')
    start_candidates = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if grid[r][c] == '.':
                # Adjacent to outer wall?
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    if (r + dr, c + dc) in outer_walls:
                        start_candidates.append((r, c))
                        break

    if not start_candidates:
        return None

    start = rng.choice(start_candidates)
    sr, sc = start
    grid[sr][sc] = 'S'

    # Place Exit on the interior island
    # Find a wall cell of the island that has an adjacent open cell inside the island
    exit_candidates = []
    for (ir, ic) in island_cells:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = ir + dr, ic + dc
            if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] == '.':
                # This open cell is inside or adjacent to the island
                # The exit should be an open cell next to the island wall
                # that is NOT reachable by wall following
                exit_candidates.append((nr, nc))

    if not exit_candidates:
        return None

    # Pick an exit that is inside the island (between island walls)
    # Prefer cells that are surrounded by more island walls
    def island_adjacency(r, c):
        count = 0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if (r + dr, c + dc) in island_cells:
                count += 1
        return count

    exit_candidates.sort(key=lambda x: -island_adjacency(x[0], x[1]))
    exit_pos = exit_candidates[0]
    er, ec = exit_pos
    grid[er][ec] = 'E'

    # Ensure the exit cell is reachable from start
    # Re-ensure connectivity (the island interior open cells must connect to
    # the outer corridor)
    ensure_open_connectivity(grid, h, w)

    # Verify: BFS finds a path
    bfs_path = bfs(grid, h, w, start, exit_pos)
    if bfs_path is None:
        return None

    # Verify: wall-following returns to start without reaching exit
    wf_trace, returned, loop_idx = wall_follow(grid, h, w, start)
    if not returned:
        return None
    if loop_idx < 25:
        return None

    # Check that wall-following trace does NOT visit the exit
    exit_in_trace = any((r, c) == exit_pos for r, c in wf_trace)
    if exit_in_trace:
        return None

    # BFS path should be non-trivial (at least 8 steps)
    if len(bfs_path) < 8:
        return None

    # Build result
    maze_display = '\n'.join(''.join(row) for row in grid)
    maze_grid = [list(row) for row in grid]

    return {
        'maze_grid': maze_grid,
        'maze_display': maze_display,
        'start': list(start),
        'exit': list(exit_pos),
        'width': w,
        'height': h,
        'wall_following_trace': [list(p) for p in wf_trace],
        'wall_following_loop_index': loop_idx,
        'wall_following_returns_to_start': True,
        'bfs_path': [list(p) for p in bfs_path],
        'bfs_path_length': len(bfs_path),
    }


# ---------------------------------------------------------------------------
# Prefill message generation
# ---------------------------------------------------------------------------

def direction_between(fr, fc, tr, tc):
    """Return direction name for a single-step move."""
    dr, dc = tr - fr, tc - fc
    if dr == -1:
        return 'up'
    if dr == 1:
        return 'down'
    if dc == -1:
        return 'left'
    if dc == 1:
        return 'right'
    return '?'


def generate_prefill(problem):
    """Generate prefill conversation messages.

    The conversation shows:
    1. User presents the maze
    2. Assistant begins wall-following (right-hand rule)
    3. Assistant passes through start again
    4. Assistant continues wall-following for at least one more step after the loop
    """
    maze_display = problem['maze_display']
    start = problem['start']
    trace = problem['wall_following_trace']
    loop_idx = problem['wall_following_loop_index']

    if loop_idx <= 0:
        return None

    # Total trace includes post-loop steps. Show all of them.
    narration_end = len(trace) - 1

    # Build move descriptions for the full trace
    # Build a compact coordinate trace showing the wall-following path.
    # Include the full loop back to start + a few steps past (the error:
    # continuing wall-following without acknowledging the loop).
    narration_end = min(len(trace) - 1, loop_idx + 4)

    # Format as coordinate chain
    coord_strs = [f"({trace[i][0]},{trace[i][1]})" for i in range(narration_end + 1)]
    trace_line = " → ".join(coord_strs)

    assistant_narration = (
        f"I'll use the right-hand rule (wall-following) to find a path.\n\n"
        f"Starting at ({start[0]},{start[1]}), following the right wall:\n"
        f"{trace_line}"
    )

    # Build prefill messages
    user_message = (
        f"Find a path from S to E in this maze. The maze uses '#' for walls "
        f"and '.' for open cells.\n\n"
        f"```\n{maze_display}\n```\n\n"
        f"Start: ({start[0]}, {start[1]})\n"
        f"Exit: ({problem['exit'][0]}, {problem['exit'][1]})\n\n"
        f"You have one response. State your answer as "
        f"SUBMIT: PATH: (r1,c1) -> (r2,c2) -> ... -> (exit_r,exit_c)"
    )

    messages = [
        {
            'role': 'user',
            'content': user_message,
        },
        {
            'role': 'assistant',
            'content': assistant_narration,
        },
    ]

    return messages


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rng = random.Random(42)
    problems = []
    attempts = 0
    max_attempts = 5000

    while len(problems) < 10 and attempts < max_attempts:
        attempts += 1
        result = generate_maze(rng)
        if result is None:
            continue

        prefill = generate_prefill(result)
        if prefill is None:
            continue

        result['prefill_messages'] = prefill
        problems.append(result)
        print(f"Generated maze {len(problems)}/10 (attempt {attempts}, "
              f"{result['height']}x{result['width']}, "
              f"wall-follow trace: {len(result['wall_following_trace'])} steps, "
              f"BFS path: {result['bfs_path_length']} steps)")

    if len(problems) < 10:
        print(f"WARNING: Only generated {len(problems)} mazes in {max_attempts} attempts")

    output = {
        'task_type': 'maze_strategy_switch',
        'description': (
            'Maze-solving task where wall-following (right-hand rule) returns to start '
            'without reaching the exit. The exit is on an interior island wall disconnected '
            'from the outer boundary. The model must recognise that wall-following has failed '
            'and switch to BFS or flood-fill.'
        ),
        'problems': problems,
    }

    output_path = Path(__file__).resolve().parent / "maze_tasks.json"
    output_path.write_text(json.dumps(output, indent=2))

    print(f"\nWrote {len(problems)} problems to {output_path.name}")

    # Validation summary
    for i, p in enumerate(problems):
        print(f"\nMaze {i+1}: {p['height']}x{p['width']}")
        print(f"  Start: {p['start']}, Exit: {p['exit']}")
        print(f"  Wall-follow trace: {len(p['wall_following_trace'])} steps, returns to start: {p['wall_following_returns_to_start']}")
        print(f"  BFS path length: {p['bfs_path_length']}")
        print(f"  Prefill messages: {len(p['prefill_messages'])}")


if __name__ == '__main__':
    main()
