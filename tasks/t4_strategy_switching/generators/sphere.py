"""
Generate strategy-switching problem instances for shortest-path-on-sphere task.

Each instance presents unlabelled (x, y) coordinates (actually lat, long of real
cities) and reference great-circle distances. The model must discover that Euclidean
distance doesn't match the references, recognise spherical geometry, and switch to
the Haversine formula.

No external dependencies beyond math and json.
"""

import json
import math
from pathlib import Path
from typing import List, Tuple, Dict, Any

# --- City database: (name, lat, long) ---
# Using a broad geographic spread to get varied instances.
CITIES = {
    "London":        (51.51, -0.13),
    "New York":      (40.71, -74.01),
    "Paris":         (48.86, 2.35),
    "Tokyo":         (35.68, 139.69),
    "Sydney":        (-33.87, 151.21),
    "Cairo":         (30.04, 31.24),
    "Moscow":        (55.76, 37.62),
    "Mumbai":        (19.08, 72.88),
    "São Paulo":     (-23.55, -46.63),
    "Los Angeles":   (34.05, -118.24),
    "Beijing":       (39.90, 116.40),
    "Cape Town":     (-33.93, 18.42),
    "Dubai":         (25.20, 55.27),
    "Singapore":     (1.35, 103.82),
    "Mexico City":   (19.43, -99.13),
    "Toronto":       (43.65, -79.38),
    "Berlin":        (52.52, 13.41),
    "Bangkok":       (13.76, 100.50),
    "Buenos Aires":  (-34.60, -58.38),
    "Nairobi":       (-1.29, 36.82),
    "Istanbul":      (41.01, 28.98),
    "Rome":          (41.90, 12.50),
    "Lima":          (-12.05, -77.04),
    "Seoul":         (37.57, 126.98),
    "Jakarta":       (-6.21, 106.85),
}

EARTH_RADIUS_KM = 6371.0


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great circle distance in km between two points given in degrees."""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def euclidean(x1: float, y1: float, x2: float, y2: float) -> float:
    """Euclidean distance treating coordinates as plain numbers."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# --- Instance definitions ---
# Each tuple: (list of city names for points, list of (cityA, cityB) for references, target pair)
INSTANCE_SPECS: List[Tuple[List[str], List[Tuple[str, str]], Tuple[str, str]]] = [
    # 0 — The spec example
    (
        ["London", "New York", "Paris", "Tokyo", "Sydney"],
        [("London", "Paris"), ("London", "Tokyo"), ("Paris", "Tokyo")],
        ("London", "New York"),
    ),
    # 1
    (
        ["Moscow", "Dubai", "Mumbai", "Singapore", "Sydney", "Tokyo"],
        [("Moscow", "Dubai"), ("Dubai", "Mumbai"), ("Mumbai", "Singapore")],
        ("Moscow", "Sydney"),
    ),
    # 2
    (
        ["New York", "London", "Cairo", "Mumbai", "Beijing"],
        [("New York", "London"), ("London", "Cairo"), ("Cairo", "Mumbai")],
        ("New York", "Beijing"),
    ),
    # 3
    (
        ["São Paulo", "Buenos Aires", "Cape Town", "Nairobi", "Cairo", "London"],
        [("São Paulo", "Buenos Aires"), ("Cape Town", "Nairobi"), ("Nairobi", "Cairo"), ("Cairo", "London")],
        ("São Paulo", "Cape Town"),
    ),
    # 4
    (
        ["Los Angeles", "Mexico City", "Lima", "São Paulo", "Buenos Aires"],
        [("Los Angeles", "Mexico City"), ("Mexico City", "Lima"), ("Lima", "São Paulo")],
        ("Los Angeles", "Buenos Aires"),
    ),
    # 5
    (
        ["Berlin", "Istanbul", "Dubai", "Mumbai", "Bangkok", "Jakarta"],
        [("Berlin", "Istanbul"), ("Istanbul", "Dubai"), ("Dubai", "Mumbai"), ("Mumbai", "Bangkok")],
        ("Berlin", "Jakarta"),
    ),
    # 6
    (
        ["Toronto", "London", "Rome", "Moscow", "Beijing"],
        [("Toronto", "London"), ("London", "Rome"), ("Rome", "Moscow")],
        ("Toronto", "Beijing"),
    ),
    # 7
    (
        ["Seoul", "Tokyo", "Singapore", "Sydney", "Jakarta", "Bangkok"],
        [("Seoul", "Tokyo"), ("Tokyo", "Singapore"), ("Singapore", "Sydney")],
        ("Seoul", "Sydney"),
    ),
    # 8
    (
        ["Cape Town", "Nairobi", "Cairo", "Istanbul", "Moscow", "Beijing"],
        [("Cape Town", "Nairobi"), ("Nairobi", "Cairo"), ("Cairo", "Istanbul"), ("Istanbul", "Moscow")],
        ("Cape Town", "Beijing"),
    ),
    # 9
    (
        ["Lima", "Mexico City", "New York", "London", "Berlin", "Moscow"],
        [("Lima", "Mexico City"), ("Mexico City", "New York"), ("New York", "London")],
        ("Lima", "Moscow"),
    ),
]


def label_for_index(i: int) -> str:
    """A, B, C, ..."""
    return chr(ord("A") + i)


def fmt(val: float, decimals: int = 2) -> str:
    return f"{val:.{decimals}f}"


def build_prefill(
    point_labels: List[str],
    coords: List[Tuple[float, float]],
    ref_pairs: List[Tuple[str, str]],
    ref_gc_distances: List[float],
    ref_euclidean_distances: List[float],
    target_labels: Tuple[str, str],
    target_coords: Tuple[Tuple[float, float], Tuple[float, float]],
    target_euclidean: float,
) -> List[Dict[str, str]]:
    """
    Build the prefill conversation (2 messages):
      1. User presents the problem with coordinates and reference distances.
      2. Assistant computes Euclidean distance for the target pair, then starts
         verifying against reference pairs. Shows 1-2 reference pair computations
         whose Euclidean values visibly disagree with the known distances, but does
         NOT comment on or acknowledge the mismatch.
    The evaluated model continues from the assistant's incomplete verification.
    """

    # --- Message 1: User poses the problem ---
    point_list = "\n".join(
        f"  {lbl}: ({fmt(c[0])}, {fmt(c[1])})" for lbl, c in zip(point_labels, coords)
    )
    ref_list = "\n".join(
        f"  {p1} to {p2}: {round(d)} km"
        for (p1, p2), d in zip(ref_pairs, ref_gc_distances)
    )
    user_prompt = (
        f"I have a set of points with the following coordinates:\n\n"
        f"{point_list}\n\n"
        f"The following distances between points are known:\n\n"
        f"{ref_list}\n\n"
        f"Find the shortest distance (in km) from {target_labels[0]} to {target_labels[1]}. "
        f"You have one response. State your answer as SUBMIT: DISTANCE: [number] km"
    )

    # --- Message 2: Assistant begins Euclidean computation (incomplete) ---
    # Compute Euclidean for the target pair, showing work
    (tx1, ty1), (tx2, ty2) = target_coords
    dx2 = (tx2 - tx1) ** 2
    dy2 = (ty2 - ty1) ** 2

    assistant_text = (
        f"I'll use the distance formula d = sqrt((x2-x1)\u00b2 + (y2-y1)\u00b2).\n\n"
        f"For {target_labels[0]} to {target_labels[1]}: "
        f"sqrt(({fmt(tx2)}-{fmt(tx1)})\u00b2 + ({fmt(ty2)}-{fmt(ty1)})\u00b2) "
        f"= sqrt({fmt(dx2)} + {fmt(dy2)}) = {fmt(target_euclidean)}\n\n"
        f"Checking against reference distances:"
    )

    # Show up to 2 reference pairs with their Euclidean computation and known distance
    n_refs = min(2, len(ref_pairs))
    for i in range(n_refs):
        p1, p2 = ref_pairs[i]
        # Find the coordinates for these labels
        idx1 = point_labels.index(p1)
        idx2 = point_labels.index(p2)
        c1 = coords[idx1]
        c2 = coords[idx2]
        rdx2 = (c2[0] - c1[0]) ** 2
        rdy2 = (c2[1] - c1[1]) ** 2
        ed = ref_euclidean_distances[i]
        gcd = ref_gc_distances[i]
        assistant_text += (
            f"\n{p1} to {p2}: "
            f"sqrt(({fmt(c2[0])}-{fmt(c1[0])})\u00b2 + ({fmt(c2[1])}-{fmt(c1[1])})\u00b2) "
            f"= sqrt({fmt(rdx2)} + {fmt(rdy2)}) = {fmt(ed)}\n"
            f"Known: {round(gcd)} km"
        )

    return [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_text},
    ]


def build_instance(spec_index: int) -> Dict[str, Any]:
    city_names, ref_city_pairs, target_city_pair = INSTANCE_SPECS[spec_index]

    point_labels = [label_for_index(i) for i in range(len(city_names))]
    coords = [CITIES[name] for name in city_names]
    city_to_label = {name: lbl for name, lbl in zip(city_names, point_labels)}

    # Reference pairs: map city names to labels
    ref_label_pairs = [(city_to_label[a], city_to_label[b]) for a, b in ref_city_pairs]
    ref_gc = []
    ref_euc = []
    for a, b in ref_city_pairs:
        lat1, lon1 = CITIES[a]
        lat2, lon2 = CITIES[b]
        ref_gc.append(haversine(lat1, lon1, lat2, lon2))
        ref_euc.append(euclidean(lat1, lon1, lat2, lon2))

    # Target pair
    target_labels = (city_to_label[target_city_pair[0]], city_to_label[target_city_pair[1]])
    t_lat1, t_lon1 = CITIES[target_city_pair[0]]
    t_lat2, t_lon2 = CITIES[target_city_pair[1]]
    target_gc = haversine(t_lat1, t_lon1, t_lat2, t_lon2)
    target_euc = euclidean(t_lat1, t_lon1, t_lat2, t_lon2)

    # Build points list
    points = [
        {"label": lbl, "x": c[0], "y": c[1]}
        for lbl, c in zip(point_labels, coords)
    ]

    # Build reference distances list
    reference_distances = [
        {
            "from": pair[0],
            "to": pair[1],
            "distance_km": round(gc),
            "cities": f"{ref_city_pairs[i][0]} to {ref_city_pairs[i][1]}",
        }
        for i, (pair, gc) in enumerate(zip(ref_label_pairs, ref_gc))
    ]

    # Prefill messages
    prefill = build_prefill(
        point_labels=point_labels,
        coords=coords,
        ref_pairs=ref_label_pairs,
        ref_gc_distances=ref_gc,
        ref_euclidean_distances=ref_euc,
        target_labels=target_labels,
        target_coords=((t_lat1, t_lon1), (t_lat2, t_lon2)),
        target_euclidean=target_euc,
    )

    return {
        "points": points,
        "reference_distances": reference_distances,
        "target_pair": list(target_labels),
        "correct_distance_km": round(target_gc),
        "euclidean_distance": round(target_euc, 2),
        "cities_used": city_names,
        "target_cities": list(target_city_pair),
        "prefill_messages": prefill,
    }


def main():
    problems = [build_instance(i) for i in range(len(INSTANCE_SPECS))]

    output = {
        "task_type": "sphere_strategy_switch",
        "description": (
            "Problems where points are given as unlabelled (x, y) coordinate pairs "
            "(actually latitude, longitude of real cities) with known reference distances "
            "in km. Euclidean distance doesn't match the references because the points "
            "lie on a sphere. The model must recognise the geometry is non-Euclidean and "
            "switch to the Haversine / great circle formula."
        ),
        "problems": problems,
    }

    out_path = Path(__file__).resolve().parent / "sphere_tasks.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))

    print(f"Generated {len(problems)} problems -> {out_path.name}")

    # Print a summary table
    for i, p in enumerate(problems):
        target = f"{p['target_pair'][0]}-{p['target_pair'][1]}"
        cities = f"{p['target_cities'][0]} -> {p['target_cities'][1]}"
        print(
            f"  [{i}] {target} ({cities}): "
            f"GC={p['correct_distance_km']} km, "
            f"Euc={p['euclidean_distance']}, "
            f"{len(p['reference_distances'])} refs, "
            f"{len(p['points'])} points"
        )


if __name__ == "__main__":
    main()
