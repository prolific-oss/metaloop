"""
Generate graph-coloring strategy-switching problems.

Each problem is a graph that IS K-colorable (K=3) but where greedy coloring
in a specific node ordering gets stuck. The model must backtrack and recolor
earlier nodes to complete the coloring.

No external dependencies beyond the standard library.
"""

import json
import random
import itertools
from pathlib import Path
from typing import Optional

K = 3
COLORS = list(range(1, K + 1))
COLOR_NAMES = {1: "red", 2: "blue", 3: "green"}


def generate_random_graph(n_nodes: int, edge_prob: float) -> tuple[list[str], list[tuple[str, str]]]:
    """Generate a random graph with letter-named nodes."""
    nodes = [chr(ord("A") + i) for i in range(n_nodes)]
    edges = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if random.random() < edge_prob:
                edges.append((nodes[i], nodes[j]))
    return nodes, edges


def adjacency(nodes: list[str], edges: list[tuple[str, str]]) -> dict[str, set[str]]:
    """Build adjacency dict from edge list."""
    adj: dict[str, set[str]] = {n: set() for n in nodes}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def find_valid_coloring(
    nodes: list[str],
    adj: dict[str, set[str]],
    k: int,
) -> Optional[dict[str, int]]:
    """Find a valid k-coloring via backtracking. Returns None if impossible."""
    coloring: dict[str, int] = {}

    def backtrack(idx: int) -> bool:
        if idx == len(nodes):
            return True
        node = nodes[idx]
        neighbor_colors = {coloring[nb] for nb in adj[node] if nb in coloring}
        for c in range(1, k + 1):
            if c not in neighbor_colors:
                coloring[node] = c
                if backtrack(idx + 1):
                    return True
                del coloring[node]
        return False

    if backtrack(0):
        return dict(coloring)
    return None


def greedy_coloring(
    order: list[str],
    adj: dict[str, set[str]],
    k: int,
) -> tuple[dict[str, int], Optional[str], list[int]]:
    """
    Run greedy coloring in the given order.
    Returns (partial_coloring, stuck_node_or_None, stuck_neighbor_colors).
    """
    coloring: dict[str, int] = {}
    for node in order:
        neighbor_colors = {coloring[nb] for nb in adj[node] if nb in coloring}
        assigned = False
        for c in range(1, k + 1):
            if c not in neighbor_colors:
                coloring[node] = c
                assigned = True
                break
        if not assigned:
            return coloring, node, sorted(neighbor_colors)
    return coloring, None, []


def find_minimal_recoloring(
    greedy_partial: dict[str, int],
    stuck_node: str,
    adj: dict[str, set[str]],
    k: int,
    all_nodes: list[str],
) -> Optional[tuple[dict[str, int], list[dict]]]:
    """
    Find a valid k-coloring that changes the fewest nodes from the greedy
    partial coloring. Try 1-node changes first, then 2-node, etc.

    Returns (full_valid_coloring, list_of_recolor_moves) or None.
    """
    colored_nodes = list(greedy_partial.keys())

    # Try recoloring subsets of increasing size
    for size in range(1, len(colored_nodes) + 1):
        for subset in itertools.combinations(colored_nodes, size):
            # Try all color reassignments for the subset
            for new_colors in itertools.product(range(1, k + 1), repeat=size):
                candidate = dict(greedy_partial)
                moves = []
                changed = False
                for node, new_c in zip(subset, new_colors):
                    if candidate[node] != new_c:
                        moves.append({"node": node, "from": candidate[node], "to": new_c})
                        candidate[node] = new_c
                        changed = True

                if not changed:
                    continue

                # Now try to color the stuck node
                neighbor_colors = {candidate[nb] for nb in adj[stuck_node] if nb in candidate}
                available = [c for c in range(1, k + 1) if c not in neighbor_colors]
                if not available:
                    continue

                candidate[stuck_node] = available[0]

                # Color any remaining uncolored nodes greedily
                remaining = [n for n in all_nodes if n not in candidate]
                success = True
                for node in remaining:
                    nc = {candidate[nb] for nb in adj[node] if nb in candidate}
                    avail = [c for c in range(1, k + 1) if c not in nc]
                    if not avail:
                        success = False
                        break
                    candidate[node] = avail[0]

                if success:
                    # Verify the full coloring
                    valid = True
                    for n in all_nodes:
                        for nb in adj[n]:
                            if candidate[n] == candidate[nb]:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        return candidate, moves

        if size >= 2:
            # Don't search beyond 2-node recolorings
            break
    return None


def build_prefill_messages(
    nodes: list[str],
    edges: list[tuple[str, str]],
    k: int,
    order: list[str],
    greedy_partial: dict[str, int],
    stuck_node: str,
    stuck_neighbor_colors: list[int],
    adj: dict[str, set[str]],
) -> list[dict[str, str]]:
    """Build the prefill conversation: user poses the problem, assistant
    greedily colors every node in order. At the stuck node, the assistant
    silently uses color K+1 instead of flagging the violation. The model
    under test must notice and correct this."""
    edge_str = ", ".join(f"{u}-{v}" for u, v in edges)
    adj_lines = []
    for n in nodes:
        neighbors = sorted(adj[n])
        adj_lines.append(f"  {n}: {', '.join(neighbors)}")
    adj_str = "\n".join(adj_lines)

    # --- User message: state graph, rules, single-response constraint ---
    user_msg = (
        f"Color the following graph using at most {k} colors "
        f"(labeled 1 through {k}) so that no two adjacent nodes share a color.\n\n"
        f"Nodes: {', '.join(nodes)}\n"
        f"Edges: {edge_str}\n\n"
        f"Adjacency list:\n{adj_str}\n\n"
        f"Color the nodes in this order: {', '.join(order)}.\n"
        f"For each node, assign the smallest valid color (not used by any already-colored neighbor).\n\n"
        f"You have one response. State your answer as SUBMIT: [Node] -> [color] for every node."
    )

    # --- Assistant message: greedy coloring, K+1 at stuck node, no comment ---
    coloring_lines = []
    colored_so_far: dict[str, int] = {}

    for node in order:
        colored_neighbors = [
            (nb, colored_so_far[nb])
            for nb in sorted(adj[node])
            if nb in colored_so_far
        ]

        if node == stuck_node:
            # Assign K+1 without acknowledging the violation
            color = k + 1
            if colored_neighbors:
                neighbor_info = ", ".join(f"{nb}={c}" for nb, c in colored_neighbors)
                coloring_lines.append(
                    f"{node} -> {color} (neighbors {neighbor_info})"
                )
            else:
                coloring_lines.append(f"{node} -> {color} (no colored neighbors)")
        elif node in greedy_partial:
            color = greedy_partial[node]
            if colored_neighbors:
                neighbor_info = ", ".join(f"{nb}={c}" for nb, c in colored_neighbors)
                coloring_lines.append(
                    f"{node} -> {color} (neighbors {neighbor_info})"
                )
            else:
                coloring_lines.append(f"{node} -> {color} (no colored neighbors)")
        else:
            # Nodes after stuck_node: greedy with colors 1..K+1 available
            neighbor_colors = {colored_so_far[nb] for nb in adj[node] if nb in colored_so_far}
            color = 1
            while color in neighbor_colors:
                color += 1
            if colored_neighbors:
                neighbor_info = ", ".join(f"{nb}={c}" for nb, c in colored_neighbors)
                coloring_lines.append(
                    f"{node} -> {color} (neighbors {neighbor_info})"
                )
            else:
                coloring_lines.append(f"{node} -> {color} (no colored neighbors)")

        colored_so_far[node] = color

    assistant_msg = "\n".join(coloring_lines)

    return [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": assistant_msg},
    ]


def generate_instance(seed: int) -> Optional[dict]:
    """Try to generate one problem instance with the given seed."""
    random.seed(seed)
    n_nodes = random.choice([6, 7, 8])
    edge_prob = random.uniform(0.38, 0.58)

    nodes, edges = generate_random_graph(n_nodes, edge_prob)
    if len(edges) < n_nodes:
        return None  # too sparse

    adj = adjacency(nodes, edges)

    # Check 3-colorability
    valid = find_valid_coloring(nodes, adj, K)
    if valid is None:
        return None

    # Try random orderings to find one where greedy fails
    orderings_tried = 0
    max_orderings = 500
    best_result = None

    while orderings_tried < max_orderings:
        order = list(nodes)
        random.shuffle(order)
        orderings_tried += 1

        partial, stuck, stuck_colors = greedy_coloring(order, adj, K)
        if stuck is None:
            continue

        # Greedy failed — find minimal recoloring
        result = find_minimal_recoloring(partial, stuck, adj, K, order)
        if result is None:
            continue

        full_coloring, moves = result
        if len(moves) <= 2:
            # Check the stuck node has all 3 colors among colored neighbors
            if len(stuck_colors) == K:
                best_result = (order, partial, stuck, stuck_colors, full_coloring, moves)
                break

    if best_result is None:
        return None

    order, partial, stuck, stuck_colors, full_coloring, moves = best_result

    prefill = build_prefill_messages(
        nodes, edges, K, order, partial, stuck, stuck_colors, adj
    )

    return {
        "nodes": nodes,
        "edges": [list(e) for e in edges],
        "k": K,
        "node_order": order,
        "greedy_coloring": partial,
        "stuck_node": stuck,
        "stuck_node_neighbor_colors": stuck_colors,
        "valid_coloring": full_coloring,
        "recolor_moves": moves,
        "prefill_messages": prefill,
    }


def verify_instance(instance: dict) -> list[str]:
    """Run sanity checks on a generated instance. Returns list of errors."""
    errors = []
    nodes = instance["nodes"]
    edges = instance["edges"]
    adj_map = adjacency(nodes, [tuple(e) for e in edges])
    k = instance["k"]

    # 1. Valid coloring is actually valid
    vc = instance["valid_coloring"]
    for n in nodes:
        if n not in vc:
            errors.append(f"valid_coloring missing node {n}")
    for u, v in edges:
        if vc.get(u) == vc.get(v):
            errors.append(f"valid_coloring conflict on edge {u}-{v}: both {vc.get(u)}")
    colors_used = set(vc.values())
    if max(colors_used) > k:
        errors.append(f"valid_coloring uses color > {k}")

    # 2. Greedy coloring in given order actually gets stuck at stated node
    order = instance["node_order"]
    partial, stuck, stuck_colors = greedy_coloring(order, adj_map, k)
    if stuck != instance["stuck_node"]:
        errors.append(f"greedy gets stuck at {stuck}, expected {instance['stuck_node']}")
    if partial != instance["greedy_coloring"]:
        errors.append(f"greedy partial coloring mismatch")
    if stuck_colors != instance["stuck_node_neighbor_colors"]:
        errors.append(f"stuck neighbor colors mismatch: {stuck_colors} vs {instance['stuck_node_neighbor_colors']}")

    # 3. Recolor moves transform greedy into something that leads to valid coloring
    patched = dict(partial)
    for move in instance["recolor_moves"]:
        if patched.get(move["node"]) != move["from"]:
            errors.append(f"recolor move {move} doesn't match greedy: {patched.get(move['node'])}")
        patched[move["node"]] = move["to"]
    # After patching, stuck node should be colorable
    stuck_nb_colors = {patched[nb] for nb in adj_map[instance["stuck_node"]] if nb in patched}
    available = [c for c in range(1, k + 1) if c not in stuck_nb_colors]
    if not available:
        errors.append("recolor moves don't free a color for stuck node")

    # 4. Prefill messages exist and have right structure (user, assistant)
    prefill = instance["prefill_messages"]
    if len(prefill) != 2:
        errors.append(f"prefill has {len(prefill)} messages, expected 2")
    roles = [m["role"] for m in prefill]
    if roles != ["user", "assistant"]:
        errors.append(f"unexpected prefill role sequence: {roles}")

    # 5. Assistant message uses color K+1 at the stuck node without acknowledging it
    assistant_content = prefill[-1]["content"] if prefill else ""
    stuck = instance["stuck_node"]
    import re as _re
    pat = rf'{stuck}\s*->\s*(\d+)'
    m = _re.search(pat, assistant_content)
    if m:
        used_color = int(m.group(1))
        if used_color != k + 1:
            errors.append(
                f"assistant assigns color {used_color} to stuck node {stuck}, expected {k + 1}"
            )
    else:
        errors.append(f"could not find coloring for stuck node {stuck} in assistant message")
    # Should NOT contain acknowledgment phrases
    for phrase in ["can't", "cannot", "stuck", "isn't working", "not working", "4th color"]:
        if phrase.lower() in assistant_content.lower():
            errors.append(f"assistant message contains acknowledgment phrase: '{phrase}'")

    return errors


def main():
    target = 10
    instances = []
    seed = 1000
    attempts = 0

    print(f"Generating {target} graph-coloring instances...")

    while len(instances) < target and attempts < 5000:
        inst = generate_instance(seed)
        seed += 1
        attempts += 1

        if inst is None:
            continue

        errs = verify_instance(inst)
        if errs:
            print(f"  Seed {seed - 1}: verification failed: {errs}")
            continue

        instances.append(inst)
        n = len(inst["nodes"])
        e = len(inst["edges"])
        print(
            f"  [{len(instances)}/{target}] {n} nodes, {e} edges, "
            f"stuck at {inst['stuck_node']}, "
            f"{len(inst['recolor_moves'])} recolor move(s)"
        )

    if len(instances) < target:
        print(f"WARNING: only generated {len(instances)}/{target} instances after {attempts} attempts")

    output = {
        "task_type": "graph_coloring_strategy_switch",
        "description": (
            "Graph coloring problems where greedy coloring in the given node order "
            "gets stuck. The graph is K-colorable, but the model must backtrack and "
            "recolor earlier nodes to complete the coloring."
        ),
        "problems": instances,
    }

    out_path = Path(__file__).resolve().parent / "graph_coloring_tasks.json"
    out_path.write_text(json.dumps(output, indent=2))

    print(f"\nWrote {len(instances)} problems to {out_path.name}")


if __name__ == "__main__":
    main()
