#!/usr/bin/env python3
"""
Generate Dijkstra-to-Bellman-Ford strategy-switching problems.

Each instance is a directed graph where Dijkstra settles a node too early
via a cheap positive path, missing a cheaper route through a negative edge
discovered later. Constructed directly (no search/rejection), verified on
generation.

Usage:
    python generate_dijkstra_tasks.py --output tasks.json
"""

import heapq
import json
import sys
from pathlib import Path

# Each spec: (nodes_str, directed_edges).
# Pattern: A→B is cheap (expanded early), B settles a node X.
# A→C is more expensive (expanded after X). C→X has a negative weight,
# giving a cheaper path to X — but Dijkstra already settled X.
# The target is the last node.

INSTANCE_SPECS = [
    # 0: 6n. Dij: A→B→D→F=7. BF: A→C→D→F=0.
    ("ABCDEF", [("A","B",1),("A","C",5),("B","D",3),("B","E",4),("C","D",-8),("D","F",3),("E","F",2)]),
    # 1: 6n. Dij: A→B→E→F=7. BF: A→C→E→F=1.
    ("ABCDEF", [("A","B",2),("A","C",6),("B","E",3),("B","D",4),("C","E",-7),("D","F",5),("E","F",2)]),
    # 2: 6n. Dij: A→B→D→F=8. BF: A→C→D→F=1.
    ("ABCDEF", [("A","B",1),("A","C",6),("B","D",4),("B","E",5),("C","D",-9),("D","F",4),("E","F",3)]),
    # 3: 7n. Dij: A→B→D→G=9. BF: A→C→D→G=2.
    ("ABCDEFG", [("A","B",1),("A","C",7),("B","D",3),("B","E",5),("C","D",-10),("D","F",2),("D","G",5),("E","G",4),("F","G",3)]),
    # 4: 6n. Dij: A→B→D→F=10. BF: A→C→D→F=-1.
    ("ABCDEF", [("A","B",2),("A","C",8),("B","D",4),("B","E",3),("C","D",-13),("D","F",4),("E","F",6)]),
    # 5: 7n. Dij: path via B. BF: path via C with negative edge.
    ("ABCDEFG", [("A","B",1),("A","C",6),("B","E",3),("B","D",2),("C","E",-9),("D","F",4),("E","G",4),("F","G",3),("D","G",7)]),
    # 6: 6n. Dij: A→B→E→F. BF: A→C→E→F with C→E negative.
    ("ABCDEF", [("A","B",1),("A","C",7),("B","E",4),("B","D",3),("C","E",-10),("D","F",6),("E","F",3)]),
    # 7: 8n. Dij: via B subtree. BF: via C with negative.
    ("ABCDEFGH", [("A","B",1),("A","C",6),("B","D",3),("B","E",2),("C","D",-11),("D","F",3),("E","G",3),("F","H",3),("G","H",4)]),
    # 8: 6n. Dij: A→B→D→F. BF: A→C→D→F.
    ("ABCDEF", [("A","B",2),("A","C",9),("B","D",5),("B","E",3),("C","D",-12),("D","F",3),("E","F",5)]),
    # 9: 7n.
    ("ABCDEFG", [("A","B",1),("A","C",8),("B","D",4),("B","E",3),("C","D",-14),("D","F",2),("D","G",6),("E","G",5),("F","G",2)]),
]


def bellman_ford(nodes, edges, source):
    dist = {n: float("inf") for n in nodes}
    prev = {n: None for n in nodes}
    dist[source] = 0
    for _ in range(len(nodes) - 1):
        for a, b, w in edges:
            if dist[a] < float("inf") and dist[a] + w < dist[b]:
                dist[b] = dist[a] + w
                prev[b] = a
    # Negative cycle check
    for a, b, w in edges:
        if dist[a] < float("inf") and dist[a] + w < dist[b]:
            raise ValueError(f"Negative cycle via {a}->{b}")
    return dist, prev


def dijkstra_solve(nodes, edges, source):
    dist = {n: float("inf") for n in nodes}
    prev = {n: None for n in nodes}
    dist[source] = 0
    visited = set()
    heap = [(0, source)]
    adj = {n: [] for n in nodes}
    for a, b, w in edges:
        adj[a].append((b, w))
    while heap:
        d, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)
        for v, w in adj[u]:
            if v not in visited and d + w < dist[v]:
                dist[v] = d + w
                prev[v] = u
                heapq.heappush(heap, (dist[v], v))
    return dist, prev


def dijkstra_query_order(nodes, edges, source):
    """Simulate Dijkstra and record ALL edge queries (including to settled nodes).

    A model running Dijkstra queries all neighbours of an expanded node,
    even if the neighbour is already settled. The settled check determines
    whether the distance is updated, not whether the edge is queried.
    """
    dist = {n: float("inf") for n in nodes}
    dist[source] = 0
    visited = set()
    heap = [(0, source)]
    order = []
    adj = {n: [] for n in nodes}
    for a, b, w in edges:
        adj[a].append((b, w))
    while heap:
        d, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)
        for v, w in adj[u]:
            order.append((u, v, w))  # query all neighbours
            if v not in visited and d + w < dist[v]:
                dist[v] = d + w
                heapq.heappush(heap, (dist[v], v))
    return order


def reconstruct_path(prev, source, target):
    path, node = [], target
    while node is not None:
        path.append(node)
        if node == source:
            return list(reversed(path))
        node = prev[node]
    return None


def build_instance(nodes_str, edges):
    nodes = list(nodes_str)
    source, target = nodes[0], nodes[-1]

    bf_dist, bf_prev = bellman_ford(nodes, edges, source)
    dj_dist, dj_prev = dijkstra_solve(nodes, edges, source)
    bf_path = reconstruct_path(bf_prev, source, target)
    dj_path = reconstruct_path(dj_prev, source, target)

    assert bf_path, f"No BF path {source}->{target}"
    assert dj_path, f"No Dijkstra path {source}->{target}"
    assert bf_dist[target] < dj_dist[target], (
        f"No divergence: BF={bf_dist[target]} DJ={dj_dist[target]}"
    )

    neg_edge = next((a, b, w) for a, b, w in edges if w < 0)

    query_order = dijkstra_query_order(nodes, edges, source)
    neg_step = None
    for i, (a, b, w) in enumerate(query_order):
        if a == neg_edge[0] and b == neg_edge[1]:
            neg_step = i + 1
            break
    assert neg_step and neg_step > 2, f"Neg edge at step {neg_step}"

    edge_list = [{"from": a, "to": b, "weight": w} for a, b, w in edges]
    topology = [{"from": a, "to": b} for a, b, _ in edges]

    # Build prefill: model uses Dijkstra, queries edges in order, encounters
    # the negative edge but continues applying Dijkstra (the error).
    prefill = build_prefill(nodes, edges, source, target, query_order, neg_step)

    return {
        "nodes": nodes, "source": source, "target": target,
        "topology": topology, "edges": edge_list,
        "negative_edge": {"from": neg_edge[0], "to": neg_edge[1], "weight": neg_edge[2]},
        "negative_edge_discovery_step": neg_step,
        "dijkstra_path": dj_path, "dijkstra_distance": dj_dist[target],
        "bellman_ford_path": bf_path, "bellman_ford_distance": bf_dist[target],
        "query_order": [{"from": a, "to": b, "weight": w} for a, b, w in query_order],
        "prefill_messages": prefill,
    }


def build_prefill(nodes, edges, source, target, query_order, neg_step):
    """Build single-turn prefill: all weights visible, Dijkstra working shown.

    The user message shows the full graph with all edge weights.
    The assistant message shows Dijkstra being applied step-by-step:
    expanding nodes, settling them, updating distances. When the negative
    edge is encountered, the assistant continues applying Dijkstra without
    reconsidering settled nodes (the error). The assistant's working ends
    mid-algorithm without declaring a final answer.
    """
    adj = {n: [] for n in nodes}
    for a, b, w in edges:
        adj[a].append((b, w))

    # Format edge list with weights
    edge_lines = [f"  {a} -> {b}: weight {w}" for a, b, w in edges]

    user_msg = (
        f"Find the shortest path from {source} to {target} in this directed graph.\n\n"
        f"Nodes: {', '.join(nodes)}\n"
        f"Edges:\n" + "\n".join(edge_lines) + "\n\n"
        f"You have one response. Show your working, then write:\n"
        f"SUBMIT: PATH: X->Y->Z DISTANCE: [number]"
    )

    # Simulate Dijkstra step-by-step for the assistant's working
    dist = {n: float("inf") for n in nodes}
    dist[source] = 0
    visited = set()
    heap = [(0, source)]
    expansion_steps = []

    while heap:
        d, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)
        neighbours = []
        for v, w in adj[u]:
            neighbours.append((v, w))
            if v not in visited and d + w < dist[v]:
                dist[v] = d + w
                heapq.heappush(heap, (dist[v], v))
        expansion_steps.append((u, d, neighbours))

    # Build the assistant's Dijkstra working
    lines = ["I'll use Dijkstra's algorithm.\n"]
    lines.append(f"Initialise: dist[{source}] = 0, all others = inf.\n")

    settled = set()
    tent_dist = {n: float("inf") for n in nodes}
    tent_dist[source] = 0

    for step_idx, (node, node_dist, neighbours) in enumerate(expansion_steps):
        settled.add(node)
        lines.append(f"Expand {node} (settled distance = {node_dist}):")

        for v, w in neighbours:
            new_dist = node_dist + w
            update = ""
            if new_dist < tent_dist[v]:
                tent_dist[v] = new_dist
                update = f" → update dist[{v}] = {new_dist}"
            else:
                update = f" → no improvement (dist[{v}] = {tent_dist[v]})"
            lines.append(f"  {node}->{v}: weight {w}, tentative {node_dist}+{w}={new_dist}{update}")

        # Show current state
        unsettled_str = ", ".join(
            f"{n}={tent_dist[n]}" for n in sorted(tent_dist)
            if n not in settled and tent_dist[n] < float("inf")
        )
        if unsettled_str:
            lines.append(f"  Unsettled: {unsettled_str}")
        lines.append("")

    # The working shows Dijkstra completing — the error is that it settled nodes
    # before the negative edge was processed, giving wrong distances.
    # Don't declare an answer — let the model continue.
    dj_target_dist = dist.get(target, "inf")
    dj_path_nodes = []
    n = target
    prev = {}
    # Reconstruct Dijkstra's path
    for step_idx, (node, node_dist, neighbours) in enumerate(expansion_steps):
        for v, w in neighbours:
            if node_dist + w == dist.get(v, float("inf")):
                prev[v] = node

    n = target
    while n:
        dj_path_nodes.append(n)
        n = prev.get(n)
    dj_path_nodes.reverse()

    lines.append(f"Dijkstra result: {' -> '.join(dj_path_nodes)}, distance = {dj_target_dist}")
    lines.append(f"\nSUBMIT: PATH: {'->'.join(dj_path_nodes)} DISTANCE: {dj_target_dist}")

    assistant_msg = "\n".join(lines)

    return [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": assistant_msg},
    ]


def main():
    default_out = Path(__file__).resolve().parent / "dijkstra_tasks.json"
    output_path = str(default_out)
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--output" and i < len(sys.argv) - 1:
            output_path = sys.argv[i + 1]

    problems = []
    for i, (nodes_str, edges) in enumerate(INSTANCE_SPECS):
        try:
            p = build_instance(nodes_str, edges)
            problems.append(p)
            print(
                f"  [{i}] {len(p['nodes'])}n  neg_step={p['negative_edge_discovery_step']}  "
                f"dj={p['dijkstra_distance']}  bf={p['bellman_ford_distance']}",
                file=sys.stderr,
            )
        except (AssertionError, ValueError) as e:
            print(f"  [{i}] FAILED: {e}", file=sys.stderr)

    output = {
        "task_type": "dijkstra_strategy_switch",
        "description": "Directed graph shortest-path problems with hidden negative edge weights.",
        "problems": problems,
    }
    print(f"Generated {len(problems)}/{len(INSTANCE_SPECS)} problems", file=sys.stderr)

    Path(output_path).write_text(json.dumps(output, indent=2))
    print(f"Saved to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
