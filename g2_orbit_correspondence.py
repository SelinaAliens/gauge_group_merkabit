"""
g2_orbit_correspondence.py
Test whether the 14 S_3-orbits of T_75 correspond to the 14 generators of G_2.

G_2 has:
  - Rank 2 (2 Cartan generators)
  - 12 roots (6 short + 6 long)
  - Root ratio: long/short = sqrt(3)
  - Dynkin diagram: two nodes, triple bond
  - dim(G_2) = 14
"""
import sys
import numpy as np
from collections import Counter, defaultdict, deque

sys.path.insert(0, '.')
from psl27_core import build_group, classify_strata, conjugacy_classes, mat_key

print("=" * 72)
print("  G_2 ORBIT CORRESPONDENCE: 14 T_75 ORBITS vs 14 G_2 GENERATORS")
print("=" * 72)

elems, e2i, mul, inv_table, ords, ID = build_group()
B_31, Z_62, T_75, z3, z3sq, W26, S3 = classify_strata(
    elems, e2i, mul, inv_table, ords, ID)
conj_cls = conjugacy_classes(mul, inv_table)

# Charge conjugation
antidiag = np.array([[0,0,1],[0,1,0],[1,0,0]], dtype=int)
C_idx = e2i[mat_key(antidiag)]
fix_C = {i for i in range(168) if mul[C_idx, mul[i, inv_table[C_idx]]] == i}

# S_3 subgroup elements
s3_involutions = [g for g in S3 if ords[g] == 2]
s3_order3 = [g for g in S3 if ords[g] == 3]

# Find generators for BFS
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

# Compute T_75 S_3-orbits
remaining = set(T_75)
orbits_t75 = []
while remaining:
    x = min(remaining)
    orbit = set()
    for g in S3:
        orbit.add(mul[g, mul[x, inv_table[g]]])
    orbits_t75.append(sorted(orbit))
    remaining -= orbit
orbits_t75.sort(key=lambda o: (len(o), o[0]))

print(f"\n  T_75 has {len(orbits_t75)} S_3-orbits")
print(f"  G_2 has 14 generators (2 Cartan + 12 roots = 2 Cartan + 6 short + 6 long)")

# Precompute distances to Z_62
dist_to_z62 = {}
for e in T_75:
    dist_to_z62[e] = bfs_distance_to_set(e, Z_62, mul, gen_set)

# =========================================================================
# 1. COMPLETE ORBIT TABLE
# =========================================================================
print("\n" + "-" * 72)
print("  1. COMPLETE T_75 ORBIT TABLE")
print("-" * 72)

orbit_data = []
for i, orb in enumerate(orbits_t75):
    orb_ords = sorted(set(int(ords[e]) for e in orb))
    n_fix_c = sum(1 for e in orb if e in fix_C)
    orb_dists = [dist_to_z62[e] for e in orb]
    mean_d = np.mean(orb_dists)

    # Conjugacy class info
    cls_indices = sorted(set(next(ci for ci, cls in enumerate(conj_cls) if e in cls) for e in orb))

    orbit_data.append({
        'idx': i, 'size': len(orb), 'orders': orb_ords,
        'fix_c': n_fix_c, 'mean_dist': mean_d,
        'cls_indices': cls_indices, 'elements': orb
    })

print(f"\n  {'Orb':>4} {'Size':>5} {'Orders':>10} {'Fix(C)':>7} {'MeanDist':>9} {'ConjCls':>10}")
print(f"  {'-'*4} {'-'*5} {'-'*10} {'-'*7} {'-'*9} {'-'*10}")
for d in orbit_data:
    print(f"  {d['idx']+1:4d} {d['size']:5d} {str(d['orders']):>10} {d['fix_c']:7d} "
          f"{d['mean_dist']:9.3f} {str(d['cls_indices']):>10}")

# =========================================================================
# 2. C-PAIRING: 6 PAIRS + 2 SELF-CONJUGATE?
# =========================================================================
print("\n" + "-" * 72)
print("  2. CHARGE CONJUGATION PAIRING OF 14 ORBITS")
print("-" * 72)

orbit_sets = [frozenset(orb) for orb in orbits_t75]
self_conj = []
paired = []
visited = set()

for i, orb in enumerate(orbits_t75):
    if i in visited:
        continue
    c_orbit = frozenset(mul[C_idx, mul[e, inv_table[C_idx]]] for e in orb)
    target = next(j for j, orb_j in enumerate(orbits_t75) if frozenset(orb_j) == c_orbit)

    if target == i:
        self_conj.append(i)
        visited.add(i)
    else:
        paired.append((i, target))
        visited.add(i)
        visited.add(target)

print(f"\n  Self-conjugate orbits: {len(self_conj)} (orbits {[i+1 for i in self_conj]})")
print(f"  C-paired orbits: {len(paired)} pairs")
for i, j in paired:
    print(f"    Orbit {i+1} (size {orbit_data[i]['size']}, orders {orbit_data[i]['orders']}) "
          f"<-> Orbit {j+1} (size {orbit_data[j]['size']}, orders {orbit_data[j]['orders']})")

print(f"\n  Decomposition: {len(self_conj)} self-conj + 2*{len(paired)} paired = "
      f"{len(self_conj) + 2*len(paired)}")
print(f"  G_2 target: 2 Cartan (self-conj) + 6 pairs (roots) = 14")
print(f"  MATCH: {len(self_conj) == 2 and len(paired) == 6}")

# Properties of self-conjugate orbits (Cartan candidates)
print(f"\n  Self-conjugate orbit details (Cartan candidates):")
for i in self_conj:
    d = orbit_data[i]
    print(f"    Orbit {i+1}: size {d['size']}, orders {d['orders']}, "
          f"Fix(C) = {d['fix_c']}, dist = {d['mean_dist']:.3f}")

# =========================================================================
# 3. TWO ROOT LENGTHS: SHORT AND LONG
# =========================================================================
print("\n" + "-" * 72)
print("  3. TWO ROOT LENGTHS: DO PAIRED ORBITS SPLIT INTO TWO CLASSES?")
print("-" * 72)

# For each paired orbit, compute several "length" metrics
print(f"\n  Paired orbit properties:")
print(f"  {'Pair':>5} {'Orbs':>10} {'Size':>5} {'Orders':>10} {'MeanDist':>9} {'ConjCls':>10}")
print(f"  {'-'*5} {'-'*10} {'-'*5} {'-'*10} {'-'*9} {'-'*10}")

pair_metrics = []
for pi, (i, j) in enumerate(paired):
    di, dj = orbit_data[i], orbit_data[j]
    # Both orbits in a pair should have same size and properties
    size = di['size']  # should equal dj['size']
    orders = di['orders']
    mean_dist = (di['mean_dist'] + dj['mean_dist']) / 2
    cls_i = di['cls_indices']
    pair_metrics.append({
        'pair_idx': pi, 'orb_i': i, 'orb_j': j,
        'size': size, 'orders': orders, 'mean_dist': mean_dist,
        'cls': cls_i
    })
    print(f"  {pi+1:5d} {f'{i+1}<->{j+1}':>10} {size:5d} {str(orders):>10} "
          f"{mean_dist:9.3f} {str(cls_i):>10}")

# Can we split the 6 pairs into 3 short + 3 long?
# Group by distance to Z_62
dist_groups = defaultdict(list)
for pm in pair_metrics:
    dist_groups[pm['mean_dist']].append(pm)

print(f"\n  Pairs grouped by distance to Z_62:")
for d, pairs in sorted(dist_groups.items()):
    print(f"    Distance {d:.3f}: {len(pairs)} pairs (orders: {[p['orders'] for p in pairs]})")

# Group by element order
order_groups = defaultdict(list)
for pm in pair_metrics:
    order_groups[tuple(pm['orders'])].append(pm)

print(f"\n  Pairs grouped by element order:")
for o, pairs in sorted(order_groups.items()):
    print(f"    Order {o}: {len(pairs)} pairs (dists: {[p['mean_dist'] for p in pairs]})")

# Test sqrt(3) ratio
if len(dist_groups) >= 2:
    dist_vals = sorted(dist_groups.keys())
    print(f"\n  Distance values: {dist_vals}")
    for i_d in range(len(dist_vals)):
        for j_d in range(i_d+1, len(dist_vals)):
            ratio = dist_vals[j_d] / dist_vals[i_d] if dist_vals[i_d] > 0 else float('inf')
            print(f"    {dist_vals[j_d]:.3f}/{dist_vals[i_d]:.3f} = {ratio:.4f} "
                  f"(sqrt(3) = {np.sqrt(3):.4f}, match: {abs(ratio - np.sqrt(3)) < 0.1})")

# =========================================================================
# 4. S_3 FIXED-POINT STRUCTURE
# =========================================================================
print("\n" + "-" * 72)
print("  4. S_3 FIXED-POINT STRUCTURE ON T_75 ORBITS")
print("-" * 72)

# For each S_3 element, which orbits are pointwise fixed?
for g in S3:
    fixed_elems = {t for t in T_75 if mul[g, mul[t, inv_table[g]]] == t}
    fixed_orbits = []
    partially_fixed = []
    for i, orb in enumerate(orbits_t75):
        n_fixed = sum(1 for e in orb if e in fixed_elems)
        if n_fixed == len(orb):
            fixed_orbits.append(i)
        elif n_fixed > 0:
            partially_fixed.append((i, n_fixed, len(orb)))

    print(f"\n  S_3 elem {g} (order {ords[g]}): {len(fixed_elems)} fixed T_75 elements")
    print(f"    Pointwise-fixed orbits: {[i+1 for i in fixed_orbits]}")
    if partially_fixed:
        print(f"    Partially-fixed orbits: {[(i+1, f'{n}/{s}') for i, n, s in partially_fixed]}")

# Orbits fixed by ALL involutions = "Cartan-like"
involution_fixed = None
for inv_elem in s3_involutions:
    fixed = {i for i, orb in enumerate(orbits_t75)
             if all(mul[inv_elem, mul[e, inv_table[inv_elem]]] == e for e in orb)}
    if involution_fixed is None:
        involution_fixed = fixed
    else:
        involution_fixed &= fixed

print(f"\n  Orbits pointwise fixed by ALL S_3 involutions: "
      f"{[i+1 for i in sorted(involution_fixed)] if involution_fixed else 'none'}")

# =========================================================================
# 5. DYNKIN DIAGRAM: 2-NODE STRUCTURE WITH TRIPLE BOND
# =========================================================================
print("\n" + "-" * 72)
print("  5. DYNKIN DIAGRAM: 2-NODE STRUCTURE WITH TRIPLE BOND?")
print("-" * 72)

# The G_2 Dynkin diagram: o===o (triple bond)
# Node 1: short roots (6 roots = 3 pairs)
# Node 2: long roots (6 roots = 3 pairs)
# The triple bond means the angle between simple roots is 5*pi/6

# Test: can we partition the 6 root-pairs into two sets of 3
# such that inter-set relationships are "triple" in some sense?

# Build an orbit adjacency graph: two orbits are adjacent if
# there exist elements a in orbit_i, b in orbit_j with a*b in T_75
print(f"\n  Orbit adjacency graph (products landing in T_75):")

orbit_adj = np.zeros((14, 14), dtype=int)
for i, orb_i in enumerate(orbits_t75):
    for j, orb_j in enumerate(orbits_t75):
        count = 0
        for a in orb_i:
            for b in orb_j:
                if mul[a, b] in T_75:
                    count += 1
        orbit_adj[i, j] = count

print(f"\n  Adjacency matrix (products within T_75):")
print(f"  {'':>4}", end='')
for j in range(14):
    print(f" {j+1:>4}", end='')
print()
for i in range(14):
    print(f"  {i+1:>4}", end='')
    for j in range(14):
        print(f" {orbit_adj[i,j]:>4}", end='')
    print()

# Normalise by orbit sizes to get coupling strength
print(f"\n  Normalised coupling (products per element pair):")
orbit_coupling = np.zeros((14, 14))
for i in range(14):
    for j in range(14):
        si, sj = len(orbits_t75[i]), len(orbits_t75[j])
        orbit_coupling[i, j] = orbit_adj[i, j] / (si * sj) if si * sj > 0 else 0

# Print coupling matrix with 3 decimal places
print(f"\n  {'':>4}", end='')
for j in range(14):
    print(f" {j+1:>5}", end='')
print()
for i in range(14):
    print(f"  {i+1:>4}", end='')
    for j in range(14):
        print(f" {orbit_coupling[i,j]:5.3f}", end='')
    print()

# Do the paired orbits cluster into two groups with stronger intra-group coupling?
# Focus on the 6 root-pair orbits (exclude self-conjugate)
if paired:
    pair_indices = []
    for i, j in paired:
        pair_indices.extend([i, j])

    # Compute average coupling within short-candidate and long-candidate groups
    # Try splitting by distance
    short_candidates = [pm for pm in pair_metrics if pm['mean_dist'] <= np.median([p['mean_dist'] for p in pair_metrics])]
    long_candidates = [pm for pm in pair_metrics if pm['mean_dist'] > np.median([p['mean_dist'] for p in pair_metrics])]

    if short_candidates and long_candidates:
        short_idxs = set()
        for pm in short_candidates:
            short_idxs.add(pm['orb_i'])
            short_idxs.add(pm['orb_j'])
        long_idxs = set()
        for pm in long_candidates:
            long_idxs.add(pm['orb_i'])
            long_idxs.add(pm['orb_j'])

        intra_short = np.mean([orbit_coupling[i, j] for i in short_idxs for j in short_idxs if i != j])
        intra_long = np.mean([orbit_coupling[i, j] for i in long_idxs for j in long_idxs if i != j])
        inter = np.mean([orbit_coupling[i, j] for i in short_idxs for j in long_idxs])

        print(f"\n  Clustering test (split by distance to Z_62):")
        print(f"    Short-root candidates: pairs {[pm['pair_idx']+1 for pm in short_candidates]}")
        print(f"    Long-root candidates: pairs {[pm['pair_idx']+1 for pm in long_candidates]}")
        print(f"    Intra-short coupling: {intra_short:.4f}")
        print(f"    Intra-long coupling:  {intra_long:.4f}")
        print(f"    Inter-group coupling:  {inter:.4f}")
        print(f"    Ratio inter/intra_short: {inter/intra_short:.4f}" if intra_short > 0 else "")

# =========================================================================
# 6. ELEMENT COUNT DECOMPOSITION
# =========================================================================
print("\n" + "-" * 72)
print("  6. ELEMENT COUNT: 75 = 2*? + 12*?")
print("-" * 72)

sc_total = sum(len(orbits_t75[i]) for i in self_conj)
paired_total = sum(len(orbits_t75[i]) + len(orbits_t75[j]) for i, j in paired)
print(f"\n  Self-conjugate elements: {sc_total}")
print(f"  Paired elements: {paired_total}")
print(f"  Total: {sc_total + paired_total}")

# For G_2: 2 Cartan + 12 roots = 14 generators
# In T_75: self-conj orbits play role of Cartan
# Paired orbits play role of root spaces
# Each root space has dimension = orbit size
print(f"\n  Mapping to G_2:")
print(f"    Cartan (self-conj): {len(self_conj)} orbits, {sc_total} elements")
print(f"    Roots (paired): {len(paired)} pairs = {2*len(paired)} orbits, {paired_total} elements")
print(f"    G_2 structure: rank 2 + 12 roots = 14")
print(f"    Our structure: {len(self_conj)} self-conj + {2*len(paired)} paired = "
      f"{len(self_conj) + 2*len(paired)}")

# =========================================================================
# SUMMARY
# =========================================================================
print("\n" + "=" * 72)
print("  SUMMARY: G_2 CORRESPONDENCE")
print("=" * 72)

match_2_6 = (len(self_conj) == 2 and len(paired) == 6)
print(f"""
  T_75 ORBIT DECOMPOSITION:
    14 S_3-orbits = {len(self_conj)} self-conjugate + {len(paired)} C-pairs
    G_2 target:   14 generators = 2 Cartan + 6 root-pairs
    STRUCTURAL MATCH (2 + 6*2 = 14): {match_2_6}

  SELF-CONJUGATE (CARTAN CANDIDATES):
    Orbits {[i+1 for i in self_conj]}
    Sizes: {[orbit_data[i]['size'] for i in self_conj]}
    Orders: {[orbit_data[i]['orders'] for i in self_conj]}

  ROOT PAIRS:
    {len(paired)} pairs, split by distance/order:
""")
for pm in sorted(pair_metrics, key=lambda x: x['mean_dist']):
    print(f"    Pair {pm['pair_idx']+1}: orbits {pm['orb_i']+1}<->{pm['orb_j']+1}, "
          f"size {pm['size']}, orders {pm['orders']}, dist {pm['mean_dist']:.3f}")
