"""Verify the Dijkstra failure pattern works."""
import heapq

def dijkstra(nodes, edges, source):
    dist = {n: float('inf') for n in nodes}
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

def bellman_ford(nodes, edges, source):
    dist = {n: float('inf') for n in nodes}
    prev = {n: None for n in nodes}
    dist[source] = 0
    for _ in range(len(nodes) - 1):
        for a, b, w in edges:
            if dist[a] < float('inf') and dist[a] + w < dist[b]:
                dist[b] = dist[a] + w
                prev[b] = a
    return dist, prev

def reconstruct(prev, source, target):
    path, node = [], target
    while node is not None:
        path.append(node)
        if node == source:
            return list(reversed(path))
        node = prev[node]
    return None

# Pattern: A→B cheap, A→C expensive, B→D positive, C→D negative.
# Dijkstra settles D via B before discovering C→D.
specs = [
    # (nodes, edges, expected_dj_F, expected_bf_F)
    ("ABCDEF", [("A","B",1),("A","C",5),("B","D",3),("B","E",4),("C","D",-8),("D","F",3),("E","F",2)]),
    ("ABCDEF", [("A","B",2),("A","C",6),("B","E",3),("B","D",4),("C","E",-7),("D","F",5),("E","F",2)]),
    ("ABCDEFG", [("A","B",1),("A","C",4),("B","D",3),("B","E",4),("C","D",-9),("D","G",5),("E","G",6),("D","F",2),("F","G",3)]),
]

for i, (nodes_str, edges) in enumerate(specs):
    nodes = list(nodes_str)
    target = nodes[-1]
    dj_dist, dj_prev = dijkstra(nodes, edges, "A")
    bf_dist, bf_prev = bellman_ford(nodes, edges, "A")
    dj_path = reconstruct(dj_prev, "A", target)
    bf_path = reconstruct(bf_prev, "A", target)
    diverges = dj_dist[target] != bf_dist[target]
    print(f"[{i}] Dijkstra {target}={dj_dist[target]} path={'->'.join(dj_path or [])}")
    print(f"    BF       {target}={bf_dist[target]} path={'->'.join(bf_path or [])}")
    print(f"    Diverges: {diverges}")
    print()
