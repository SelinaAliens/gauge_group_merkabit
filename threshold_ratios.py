"""
threshold_ratios.py
Test whether the internal thresholds of the three strata have ratios
consistent with h*(E6):h*(B6):h*(C6) = 12:11:7.

Questions:
1. Do weighted mean BFS depths D(B), D(Z), D(T) form a ratio ~ 12:11:7?
2. Does the orbit count 8+11+14 = 33 have a Coxeter/Langlands interpretation?
3. Is the threshold a single thing or a spectrum?
"""
import sys
import numpy as np
from collections import Counter, defaultdict, deque
from itertools import combinations

sys.path.insert(0, '.')
from psl27_core import build_group, classify_strata, mat_key

print("=" * 72)
print("  THRESHOLD RATIOS: h*(E6) : h*(B6) : h*(C6) = 12 : 11 : 7")
print("=" * 72)

elems, e2i, mul, inv_table, ords, ID = build_group()
B_31, Z_62, T_75, z3, z3sq, W26, S3 = classify_strata(
    elems, e2i, mul, inv_table, ords, ID)

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

# Compute S_3 orbits
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

# =========================================================================
# 1. THRESHOLD DEPTHS — MULTIPLE DEFINITIONS
# =========================================================================
print("\n" + "-" * 72)
print("  1. THRESHOLD DEPTH COMPUTATIONS")
print("-" * 72)

print("\n  Computing BFS distances from all elements to Z_62...")

# BFS distance from every element to Z_62
dist_to_z62 = {}
for e in range(168):
    if e in Z_62:
        dist_to_z62[e] = 0
    else:
        dist_to_z62[e] = bfs_distance_to_set(e, Z_62, mul, gen_set)

# BFS distance from Z_62 elements to W26
dist_z62_to_w26 = {}
for e in sorted(Z_62):
    dist_z62_to_w26[e] = bfs_distance_to_set(e, W26, mul, gen_set)

print("  Done.")

# Definition 1: Unweighted mean distance to Z_62
D_B_unweighted = np.mean([dist_to_z62[e] for e in B_31])
D_Z_internal = np.mean([dist_z62_to_w26[e] for e in Z_62])
D_T_unweighted = np.mean([dist_to_z62[e] for e in T_75])

print(f"\n  Definition 1: Unweighted mean BFS distance to Z_62")
print(f"    D(B_31) = {D_B_unweighted:.6f}")
print(f"    D(Z_62 -> W26) = {D_Z_internal:.6f}")
print(f"    D(T_75) = {D_T_unweighted:.6f}")

# Definition 2: Weighted by S_3-orbit size
def orbit_weighted_depth(orbits, dist_dict):
    total_weight = 0
    weighted_sum = 0
    for orb in orbits:
        orb_mean_dist = np.mean([dist_dict[e] for e in orb])
        weighted_sum += len(orb) * orb_mean_dist
        total_weight += len(orb)
    return weighted_sum / total_weight

D_B_weighted = orbit_weighted_depth(orbits_b31, dist_to_z62)
D_Z_weighted = orbit_weighted_depth(orbits_z62, dist_z62_to_w26)
D_T_weighted = orbit_weighted_depth(orbits_t75, dist_to_z62)

print(f"\n  Definition 2: S_3-orbit-weighted mean BFS distance")
print(f"    D(B_31) = {D_B_weighted:.6f}")
print(f"    D(Z_62) = {D_Z_weighted:.6f}")
print(f"    D(T_75) = {D_T_weighted:.6f}")

# Definition 3: Median distance
D_B_median = np.median([dist_to_z62[e] for e in B_31])
D_Z_median = np.median([dist_z62_to_w26[e] for e in Z_62])
D_T_median = np.median([dist_to_z62[e] for e in T_75])

print(f"\n  Definition 3: Median BFS distance")
print(f"    D(B_31) = {D_B_median:.1f}")
print(f"    D(Z_62) = {D_Z_median:.1f}")
print(f"    D(T_75) = {D_T_median:.1f}")

# =========================================================================
# 2. RATIO TESTS
# =========================================================================
print("\n" + "-" * 72)
print("  2. RATIO TESTS: D(B) : D(Z) : D(T) vs 12 : 11 : 7")
print("-" * 72)

h_dual = {'E6': 12, 'B6': 11, 'C6': 7}
target_ratio = np.array([12, 11, 7])

def test_ratio(D_B, D_Z, D_T, label):
    depths = np.array([D_B, D_Z, D_T])
    if min(depths) == 0:
        print(f"\n  {label}: CANNOT TEST (contains zero depth)")
        return

    # Normalise to sum=30 (same as 12+11+7)
    normalised = depths * 30.0 / depths.sum()
    residual = np.linalg.norm(normalised - target_ratio)

    # Also try other orderings: B↔E6, Z↔B6, T↔C6
    print(f"\n  {label}:")
    print(f"    Raw depths: B={D_B:.4f}, Z={D_Z:.4f}, T={D_T:.4f}")
    print(f"    Normalised (sum=30): B={normalised[0]:.3f}, Z={normalised[1]:.3f}, T={normalised[2]:.3f}")
    print(f"    Target (12:11:7):    B=12.000, Z=11.000, T=7.000")
    print(f"    L2 residual: {residual:.4f}")

    # Try all 6 permutations of assignment
    from itertools import permutations
    best_perm = None
    best_resid = float('inf')
    for perm in permutations([0, 1, 2]):
        perm_depths = np.array([depths[perm[0]], depths[perm[1]], depths[perm[2]]])
        perm_norm = perm_depths * 30.0 / perm_depths.sum()
        resid = np.linalg.norm(perm_norm - target_ratio)
        if resid < best_resid:
            best_resid = resid
            best_perm = perm

    labels = ['B', 'Z', 'T']
    algebras = ['E6(h*=12)', 'B6(h*=11)', 'C6(h*=7)']
    print(f"\n    Best permutation (min residual {best_resid:.4f}):")
    for j, alg in enumerate(algebras):
        print(f"      {alg} <-> {labels[best_perm[j]]} (depth={depths[best_perm[j]]:.4f})")

    # Ratios between pairs
    print(f"\n    Pairwise ratios:")
    print(f"      D(B)/D(T) = {D_B/D_T:.4f}  (12/7 = {12/7:.4f})")
    print(f"      D(B)/D(Z) = {D_B/D_Z:.4f}  (12/11 = {12/11:.4f})" if D_Z > 0 else "")
    print(f"      D(T)/D(Z) = {D_T/D_Z:.4f}  (7/11 = {7/11:.4f})" if D_Z > 0 else "")

test_ratio(D_B_unweighted, D_Z_internal, D_T_unweighted, "Unweighted mean")
test_ratio(D_B_weighted, D_Z_weighted, D_T_weighted, "S3-orbit weighted")

# =========================================================================
# 3. ALTERNATIVE: DEPTHS AS ORBIT-COUNT FRACTIONS
# =========================================================================
print("\n" + "-" * 72)
print("  3. ORBIT-COUNT INTERPRETATION: 8 + 11 + 14 = 33")
print("-" * 72)

n_orbs = np.array([len(orbits_b31), len(orbits_z62), len(orbits_t75)])
print(f"\n  S_3-orbit counts: B={n_orbs[0]}, Z={n_orbs[1]}, T={n_orbs[2]}")
print(f"  Sum: {n_orbs.sum()}")

# 33 = dim(E6 adjoint) - rank = 78 - 6 = 72?  No, 33 != 72.
# 33 in Coxeter theory:
print(f"\n  Coxeter/Langlands interpretations of 33:")
print(f"    dim(E6) - dim(F4) = 78 - 52 = 26 (no)")
print(f"    dim(SO(11)) - dim(SO(8)) = 55 - 28 = 27 (no)")
print(f"    Number of positive roots of B_5 = 5*4/2 + 5 = 15 (no)")
print(f"    33 = 3 * 11 = 3 * h*(B6)")
print(f"    33 = |W(A2)| * 11/2? No, |W(A2)| = 6")
print(f"    8 = dim(SU(3) adj), 14 = dim(G2 adj)")
print(f"    11 = h*(B6) = h(B6) - 1 = 12 - 1")

# Orbit count ratios vs h*
print(f"\n  Orbit count ratios:")
print(f"    B/Z = {n_orbs[0]/n_orbs[1]:.4f}  (12/11 = {12/11:.4f})")
print(f"    B/T = {n_orbs[0]/n_orbs[2]:.4f}  (12/7  = {12/7:.4f})")
print(f"    Z/T = {n_orbs[1]/n_orbs[2]:.4f}  (11/7  = {11/7:.4f})")

# Try inverse: T has most orbits = deepest = smallest h*?
print(f"\n  Inverse orbit ratios (T=deepest -> smallest h*):")
print(f"    T/Z = {n_orbs[2]/n_orbs[1]:.4f}  (7/11 = {7/11:.4f}, no)")
print(f"    T/B = {n_orbs[2]/n_orbs[0]:.4f}  (7/12 = {7/12:.4f}, no)")

# Test: h* proportional to inverse orbit count?
# 12:11:7 vs 1/8 : 1/11 : 1/14 = 0.125 : 0.0909 : 0.0714
inv_orbs = 1.0 / n_orbs
inv_norm = inv_orbs * 30.0 / inv_orbs.sum()
print(f"\n  Inverse orbit counts normalised (sum=30):")
print(f"    B={inv_norm[0]:.3f}, Z={inv_norm[1]:.3f}, T={inv_norm[2]:.3f}")
print(f"    Target: 12.000, 11.000, 7.000")
print(f"    Residual: {np.linalg.norm(inv_norm - target_ratio):.4f}")

# =========================================================================
# 4. ALTERNATIVE METRICS: MAX DISTANCE, SPREAD, ENTROPY
# =========================================================================
print("\n" + "-" * 72)
print("  4. ALTERNATIVE THRESHOLD METRICS")
print("-" * 72)

# Max distance per stratum
max_B = max(dist_to_z62[e] for e in B_31)
max_T = max(dist_to_z62[e] for e in T_75)
max_Z = max(dist_z62_to_w26[e] for e in Z_62)

print(f"\n  Max distances:")
print(f"    max(B -> Z_62) = {max_B}")
print(f"    max(Z -> W26)  = {max_Z}")
print(f"    max(T -> Z_62) = {max_T}")

if max_Z > 0:
    print(f"\n  Max distance ratios:")
    print(f"    max_B/max_Z = {max_B/max_Z:.4f}  (12/11 = {12/11:.4f})")
    print(f"    max_T/max_Z = {max_T/max_Z:.4f}  (7/11  = {7/11:.4f})")
    print(f"    max_B/max_T = {max_B/max_T:.4f}  (12/7  = {12/7:.4f})")

# Distance entropy per stratum
def entropy(values):
    counts = Counter(values)
    total = sum(counts.values())
    probs = [c/total for c in counts.values()]
    return -sum(p * np.log2(p) for p in probs if p > 0)

H_B = entropy([dist_to_z62[e] for e in B_31])
H_Z = entropy([dist_z62_to_w26[e] for e in Z_62])
H_T = entropy([dist_to_z62[e] for e in T_75])

print(f"\n  Distance distribution entropy (bits):")
print(f"    H(B) = {H_B:.4f}")
print(f"    H(Z) = {H_Z:.4f}")
print(f"    H(T) = {H_T:.4f}")

# =========================================================================
# 5. STRATUM-TO-STRATUM PRODUCT DISTANCE
# =========================================================================
print("\n" + "-" * 72)
print("  5. PRODUCT DISTANCES (NUMBER OF MULTIPLICATIONS)")
print("-" * 72)

# For each element of B_31, how many elements of PSL(2,7) map it into Z_62
# by a single right-multiplication?
b_to_z_1step = {}
for e in sorted(B_31):
    count = sum(1 for g in range(168) if mul[e, g] in Z_62)
    b_to_z_1step[e] = count

print(f"\n  B_31 -> Z_62 in one right-multiplication:")
print(f"    Mean multipliers per element: {np.mean(list(b_to_z_1step.values())):.1f}")
print(f"    Min:  {min(b_to_z_1step.values())}")
print(f"    Max:  {max(b_to_z_1step.values())}")
print(f"    Distribution: {dict(Counter(b_to_z_1step.values()))}")

t_to_z_1step = {}
for e in sorted(T_75):
    count = sum(1 for g in range(168) if mul[e, g] in Z_62)
    t_to_z_1step[e] = count

print(f"\n  T_75 -> Z_62 in one right-multiplication:")
print(f"    Mean multipliers per element: {np.mean(list(t_to_z_1step.values())):.1f}")
print(f"    Min:  {min(t_to_z_1step.values())}")
print(f"    Max:  {max(t_to_z_1step.values())}")
print(f"    Distribution: {dict(Counter(t_to_z_1step.values()))}")

# Product distance = |Z_62| * |G| / (number of right-multipliers landing in Z_62)
# This is a "resistance" metric
print(f"\n  Accessibility ratio (fraction of right-multipliers landing in Z_62):")
b_access = np.mean(list(b_to_z_1step.values())) / 168
t_access = np.mean(list(t_to_z_1step.values())) / 168
print(f"    B_31 accessibility: {b_access:.4f}")
print(f"    T_75 accessibility: {t_access:.4f}")
print(f"    Expected (uniform): {len(Z_62)/168:.4f} = {len(Z_62)}/168")
print(f"    B/T accessibility ratio: {b_access/t_access:.4f}")

# =========================================================================
# 6. THE MASS GAP: Δ = 1/24 INTERPRETATION
# =========================================================================
print("\n" + "-" * 72)
print("  6. MASS GAP AND ORBIT-STABILISER")
print("-" * 72)

# For each S3-orbit, compute the stabiliser (elements of S3 fixing every element)
print(f"\n  Orbit stabiliser sizes:")

for label, orbits in [('B_31', orbits_b31), ('Z_62', orbits_z62), ('T_75', orbits_t75)]:
    print(f"\n  {label}:")
    for i, orb in enumerate(orbits):
        stab_size = 0
        for g in S3:
            if all(mul[g, mul[e, inv_table[g]]] == e for e in orb):
                stab_size += 1
        print(f"    Orbit {i+1} (size {len(orb)}): stabiliser size = {stab_size}, "
              f"|S3|/|orb| = {len(S3)/len(orb):.2f}")

# Δ = 1/24 as orbit-stabiliser invariant
# 1/24 = 1/(|S3| * |Z4|) where Z4 is the bridge cyclic group
print(f"\n  Delta = 1/24 decompositions:")
print(f"    1/24 = 1/(|S3|*|Z4|) = 1/(6*4) = 1/24")
print(f"    1/24 = 1/(h(E6)*2) = 1/(12*2) = 1/24")

# =========================================================================
# 7. h* SUM TEST: 12 + 11 + 7 = 30 = h*(E8)
# =========================================================================
print("\n" + "-" * 72)
print("  7. h* SUM: 12 + 11 + 7 = 30 = h*(E8)?")
print("-" * 72)

print(f"\n  Dual Coxeter numbers:")
print(f"    h*(E6) = 12")
print(f"    h*(B6) = 11")
print(f"    h*(C6) = 7")
print(f"    Sum = 30")
print(f"\n  h*(E8) = 30")
print(f"  h*(D16) = 30")
print(f"  h*(B15) = 29 (no)")
print(f"\n  Is 30 = h*(E8)? YES")
print(f"  Interpretation: The three stratum thresholds sum to the")
print(f"  dual Coxeter number of E8, the largest exceptional algebra.")

# =========================================================================
# SUMMARY
# =========================================================================
print("\n" + "=" * 72)
print("  SUMMARY: THRESHOLD RATIOS")
print("=" * 72)

print(f"""
  THRESHOLD DEPTHS (unweighted mean BFS):
    D(B_31 -> Z_62) = {D_B_unweighted:.4f}
    D(Z_62 -> W26)  = {D_Z_internal:.4f}
    D(T_75 -> Z_62) = {D_T_unweighted:.4f}

  THRESHOLD DEPTHS (S3-orbit weighted):
    D(B_31) = {D_B_weighted:.4f}
    D(Z_62) = {D_Z_weighted:.4f}
    D(T_75) = {D_T_weighted:.4f}

  ORBIT COUNTS: 8 + 11 + 14 = 33
    8 = dim(SU(3) adjoint)
    14 = dim(G2 adjoint)
    11 = h*(B6)
    33 = 3 * 11

  h* PREDICTIONS:
    12 : 11 : 7  (E6 : B6 : C6)
    Sum = 30 = h*(E8)

  QUESTION ANSWERED:
    Is the threshold a single thing or a spectrum?
    -> Answer: [see distance distributions above]
""")
