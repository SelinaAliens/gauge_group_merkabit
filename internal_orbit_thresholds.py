"""
internal_orbit_thresholds.py
For each stratum, identify the internal threshold — the S_3-orbit boundary
that separates "threshold-adjacent" elements from "interior" elements.

Questions:
1. B_31: Do Fix(C) orbits (size-3) have shorter distance to Z_62 than size-6 orbits?
2. Z_62: How many orbits are purely Weinberg, purely non-Weinberg, or mixed?
3. T_75: Do order-4 bridge elements sit in the closest orbits to Z_62?
4. Does the orbit distance gradient align with the force hierarchy?
"""
import sys
import numpy as np
from collections import Counter, defaultdict, deque

sys.path.insert(0, '.')
from psl27_core import build_group, classify_strata, mat_key

print("=" * 72)
print("  INTERNAL ORBIT THRESHOLDS")
print("=" * 72)

elems, e2i, mul, inv_table, ords, ID = build_group()
B_31, Z_62, T_75, z3, z3sq, W26, S3 = classify_strata(
    elems, e2i, mul, inv_table, ords, ID)

# Charge conjugation
antidiag = np.array([[0,0,1],[0,1,0],[1,0,0]], dtype=int)
C_idx = e2i[mat_key(antidiag)]
fix_C = {i for i in range(168) if mul[C_idx, mul[i, inv_table[C_idx]]] == i}

# Find generators
def find_generators(mul_table, ords_arr, ID_elem):
    ord2 = [i for i in range(168) if ords_arr[i] == 2]
    ord3 = [i for i in range(168) if ords_arr[i] == 3]
    for a in ord2:
        for b in ord3[:10]:
            generated = {ID_elem}
            frontier = {ID_elem, a, b}
            while frontier - generated:
                generated.update(frontier)
                new = set()
                for x in frontier:
                    for g in [a, b]:
                        new.add(mul_table[x, g])
                        new.add(mul_table[g, x])
                frontier = new - generated
            if len(generated) == 168:
                return [a, b]
    return None

gen_pair = find_generators(mul, ords, ID)
gen_set = sorted(set(gen_pair + [inv_table[g] for g in gen_pair]))
print(f"  Generators: {gen_pair} (orders {[int(ords[g]) for g in gen_pair]})")

def bfs_distance_to_set(start, target_set, mul_table, gens):
    if start in target_set:
        return 0
    visited = {start}
    queue = deque([(start, 0)])
    while queue:
        current, dist = queue.popleft()
        for g in gens:
            for nxt in [mul_table[current, g], mul_table[g, current]]:
                if nxt in target_set:
                    return dist + 1
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, dist + 1))
    return float('inf')

# Compute S_3 orbits for each stratum
def compute_s3_orbits(stratum):
    remaining = set(stratum)
    orbits = []
    while remaining:
        x = min(remaining)
        orbit = set()
        for g in S3:
            orbit.add(mul[g, mul[x, inv_table[g]]])
        orbits.append(sorted(orbit))
        remaining -= orbit
    orbits.sort(key=lambda o: (len(o), o[0]))
    return orbits

orbits_b31 = compute_s3_orbits(B_31)
orbits_z62 = compute_s3_orbits(Z_62)
orbits_t75 = compute_s3_orbits(T_75)

print(f"  S3-orbits: B_31={len(orbits_b31)}, Z_62={len(orbits_z62)}, T_75={len(orbits_t75)}")

# Precompute all BFS distances to Z_62 for B_31 and T_75
print("\n  Computing BFS distances...")
dist_to_z62 = {}
for e in sorted(B_31 | T_75):
    dist_to_z62[e] = bfs_distance_to_set(e, Z_62, mul, gen_set)
for e in Z_62:
    dist_to_z62[e] = 0

# Precompute Z_62 internal distances to W26
dist_to_w26 = {}
for e in sorted(Z_62):
    dist_to_w26[e] = bfs_distance_to_set(e, W26, mul, gen_set)

print("  Done.")

# =========================================================================
# 1. B_31 ORBIT ANALYSIS
# =========================================================================
print("\n" + "-" * 72)
print("  1. B_31: 8 S_3-ORBITS RANKED BY THRESHOLD PROXIMITY")
print("-" * 72)

print(f"\n  {'Orb':>4} {'Size':>5} {'Orders':>10} {'Fix(C)':>7} {'MeanDist':>9} "
      f"{'MinDist':>8} {'MaxDist':>8} {'Type'}")
print(f"  {'-'*4} {'-'*5} {'-'*10} {'-'*7} {'-'*9} {'-'*8} {'-'*8} {'-'*20}")

b31_orbit_data = []
for i, orb in enumerate(orbits_b31):
    orb_ords = sorted(set(int(ords[e]) for e in orb))
    n_fix_c = sum(1 for e in orb if e in fix_C)
    orb_dists = [dist_to_z62[e] for e in orb]
    mean_d = np.mean(orb_dists)
    min_d = min(orb_dists)
    max_d = max(orb_dists)

    # Classify: threshold-adjacent (has Fix(C)) vs interior
    otype = "THRESHOLD (Fix(C))" if n_fix_c > 0 else "INTERIOR"

    b31_orbit_data.append({
        'idx': i, 'size': len(orb), 'orders': orb_ords,
        'fix_c': n_fix_c, 'mean_dist': mean_d, 'min_dist': min_d,
        'max_dist': max_d, 'type': otype, 'elements': orb
    })

    print(f"  {i+1:4d} {len(orb):5d} {str(orb_ords):>10} {n_fix_c:7d} {mean_d:9.3f} "
          f"{min_d:8d} {max_d:8d} {otype}")

# Test: Do Fix(C) orbits have shorter distance?
fix_c_dists = [d['mean_dist'] for d in b31_orbit_data if d['fix_c'] > 0]
no_fix_c_dists = [d['mean_dist'] for d in b31_orbit_data if d['fix_c'] == 0]
print(f"\n  Fix(C) orbits mean distance to Z_62: {np.mean(fix_c_dists):.3f} (n={len(fix_c_dists)})")
print(f"  No-Fix(C) orbits mean distance:      {np.mean(no_fix_c_dists):.3f} (n={len(no_fix_c_dists)})")
print(f"  Fix(C) orbits CLOSER? {np.mean(fix_c_dists) < np.mean(no_fix_c_dists)}")

# Size-3 vs size-6 orbits
size3_dists = [d['mean_dist'] for d in b31_orbit_data if d['size'] == 3]
size6_dists = [d['mean_dist'] for d in b31_orbit_data if d['size'] == 6]
size1_dists = [d['mean_dist'] for d in b31_orbit_data if d['size'] == 1]
print(f"\n  Size-1 orbits (identity): mean dist = {np.mean(size1_dists):.3f}" if size1_dists else "")
if size3_dists:
    print(f"  Size-3 orbits: mean dist = {np.mean(size3_dists):.3f} (n={len(size3_dists)})")
if size6_dists:
    print(f"  Size-6 orbits: mean dist = {np.mean(size6_dists):.3f} (n={len(size6_dists)})")
if size3_dists and size6_dists:
    print(f"  Size-3 CLOSER than size-6? {np.mean(size3_dists) < np.mean(size6_dists)}")

# =========================================================================
# 2. Z_62 ORBIT ANALYSIS — WEINBERG BOUNDARY
# =========================================================================
print("\n" + "-" * 72)
print("  2. Z_62: 11 S_3-ORBITS — WEINBERG/NON-WEINBERG CLASSIFICATION")
print("-" * 72)

print(f"\n  {'Orb':>4} {'Size':>5} {'Orders':>10} {'W26':>4} {'nonW':>5} "
      f"{'MeanDist':>9} {'Type'}")
print(f"  {'-'*4} {'-'*5} {'-'*10} {'-'*4} {'-'*5} {'-'*9} {'-'*25}")

z62_orbit_data = []
n_pure_weinberg = 0
n_pure_non_weinberg = 0
n_mixed = 0

for i, orb in enumerate(orbits_z62):
    orb_ords = sorted(set(int(ords[e]) for e in orb))
    n_w = sum(1 for e in orb if e in W26)
    n_nw = len(orb) - n_w
    orb_dists = [dist_to_w26[e] for e in orb]
    mean_d = np.mean(orb_dists)

    if n_w == len(orb):
        otype = "PURELY WEINBERG"
        n_pure_weinberg += 1
    elif n_w == 0:
        otype = "PURELY NON-WEINBERG"
        n_pure_non_weinberg += 1
    else:
        otype = "MIXED"
        n_mixed += 1

    z62_orbit_data.append({
        'idx': i, 'size': len(orb), 'orders': orb_ords,
        'n_weinberg': n_w, 'n_non_weinberg': n_nw,
        'mean_dist': mean_d, 'type': otype, 'elements': orb
    })

    print(f"  {i+1:4d} {len(orb):5d} {str(orb_ords):>10} {n_w:4d} {n_nw:5d} "
          f"{mean_d:9.3f} {otype}")

print(f"\n  Purely Weinberg orbits:     {n_pure_weinberg}")
print(f"  Purely non-Weinberg orbits: {n_pure_non_weinberg}")
print(f"  Mixed orbits:               {n_mixed}")

# The 36 non-Weinberg elements
non_weinberg_z62 = Z_62 - W26
print(f"\n  Non-Weinberg Z_62 elements: {len(non_weinberg_z62)}")
nw_orders = Counter(int(ords[e]) for e in non_weinberg_z62)
print(f"  Non-Weinberg order distribution: {dict(sorted(nw_orders.items()))}")
w_orders = Counter(int(ords[e]) for e in W26)
print(f"  Weinberg order distribution: {dict(sorted(w_orders.items()))}")

# =========================================================================
# 3. T_75 ORBIT ANALYSIS — CONFINEMENT GRADIENT
# =========================================================================
print("\n" + "-" * 72)
print("  3. T_75: 14 S_3-ORBITS RANKED BY DISTANCE TO Z_62")
print("-" * 72)

print(f"\n  {'Orb':>4} {'Size':>5} {'Orders':>10} {'Fix(C)':>7} {'MeanDist':>9} "
      f"{'MinDist':>8} {'MaxDist':>8}")
print(f"  {'-'*4} {'-'*5} {'-'*10} {'-'*7} {'-'*9} {'-'*8} {'-'*8}")

t75_orbit_data = []
for i, orb in enumerate(orbits_t75):
    orb_ords = sorted(set(int(ords[e]) for e in orb))
    n_fix_c = sum(1 for e in orb if e in fix_C)
    orb_dists = [dist_to_z62[e] for e in orb]
    mean_d = np.mean(orb_dists)
    min_d = min(orb_dists)
    max_d = max(orb_dists)

    t75_orbit_data.append({
        'idx': i, 'size': len(orb), 'orders': orb_ords,
        'fix_c': n_fix_c, 'mean_dist': mean_d, 'min_dist': min_d,
        'max_dist': max_d, 'elements': orb
    })

    print(f"  {i+1:4d} {len(orb):5d} {str(orb_ords):>10} {n_fix_c:7d} {mean_d:9.3f} "
          f"{min_d:8d} {max_d:8d}")

# Sort orbits by mean distance (threshold proximity ranking)
t75_sorted = sorted(t75_orbit_data, key=lambda d: d['mean_dist'])
print(f"\n  T_75 orbits ranked by threshold proximity (closest first):")
for rank, d in enumerate(t75_sorted):
    print(f"    Rank {rank+1}: Orbit {d['idx']+1} (size {d['size']}, "
          f"orders {d['orders']}, mean dist {d['mean_dist']:.3f})")

# Are order-4 elements closest?
order4_orbits = [d for d in t75_orbit_data if 4 in d['orders']]
non_order4_orbits = [d for d in t75_orbit_data if 4 not in d['orders']]
if order4_orbits:
    o4_mean = np.mean([d['mean_dist'] for d in order4_orbits])
    no4_mean = np.mean([d['mean_dist'] for d in non_order4_orbits])
    print(f"\n  Order-4 containing orbits: mean dist = {o4_mean:.3f} (n={len(order4_orbits)})")
    print(f"  Non-order-4 orbits:        mean dist = {no4_mean:.3f} (n={len(non_order4_orbits)})")
    print(f"  Order-4 orbits CLOSER? {o4_mean < no4_mean}")

# Order-7 elements (Fano core) — are they the deepest?
order7_orbits = [d for d in t75_orbit_data if 7 in d['orders']]
if order7_orbits:
    o7_mean = np.mean([d['mean_dist'] for d in order7_orbits])
    print(f"\n  Order-7 containing orbits: mean dist = {o7_mean:.3f} (n={len(order7_orbits)})")
    print(f"  Order-7 orbits DEEPEST? {o7_mean >= np.mean([d['mean_dist'] for d in t75_orbit_data])}")

# =========================================================================
# 4. COMPLETE THRESHOLD PROXIMITY TABLE
# =========================================================================
print("\n" + "-" * 72)
print("  4. COMPLETE ORBIT THRESHOLD-PROXIMITY TABLE")
print("-" * 72)

all_orbits = []
for d in b31_orbit_data:
    all_orbits.append(('B', d['idx']+1, d['size'], d['orders'], d['mean_dist']))
for d in z62_orbit_data:
    all_orbits.append(('Z', d['idx']+1, d['size'], d['orders'], d['mean_dist']))
for d in t75_orbit_data:
    all_orbits.append(('T', d['idx']+1, d['size'], d['orders'], d['mean_dist']))

# Sort by mean distance to threshold
all_orbits.sort(key=lambda x: x[4])

print(f"\n  {'Stratum':>8} {'Orb':>4} {'Size':>5} {'Orders':>12} {'MeanDist':>9}")
print(f"  {'-'*8} {'-'*4} {'-'*5} {'-'*12} {'-'*9}")
for stratum, oidx, size, orders, mean_d in all_orbits:
    print(f"  {stratum:>8} {oidx:4d} {size:5d} {str(orders):>12} {mean_d:9.3f}")

# =========================================================================
# SUMMARY
# =========================================================================
print("\n" + "=" * 72)
print("  SUMMARY: INTERNAL ORBIT THRESHOLDS")
print("=" * 72)

b_threshold = [d for d in b31_orbit_data if d['fix_c'] > 0]
b_interior = [d for d in b31_orbit_data if d['fix_c'] == 0]
z_threshold = [d for d in z62_orbit_data if d['n_weinberg'] > 0]
z_interior = [d for d in z62_orbit_data if d['n_weinberg'] == 0]

print(f"""
  B_31 internal boundary:
    Threshold-adjacent (Fix(C)): {len(b_threshold)} orbits, {sum(d['size'] for d in b_threshold)} elements
    Interior (no Fix(C)):        {len(b_interior)} orbits, {sum(d['size'] for d in b_interior)} elements

  Z_62 internal boundary:
    Weinberg-containing:  {len(z_threshold)} orbits
    Non-Weinberg:         {len(z_interior)} orbits
    The 36 non-Weinberg elements: orders = {dict(sorted(nw_orders.items()))}

  T_75 gradient:
    Closest orbit to Z_62: Orbit {t75_sorted[0]['idx']+1} (size {t75_sorted[0]['size']}, orders {t75_sorted[0]['orders']}, dist {t75_sorted[0]['mean_dist']:.3f})
    Deepest orbit from Z_62: Orbit {t75_sorted[-1]['idx']+1} (size {t75_sorted[-1]['size']}, orders {t75_sorted[-1]['orders']}, dist {t75_sorted[-1]['mean_dist']:.3f})
""")
