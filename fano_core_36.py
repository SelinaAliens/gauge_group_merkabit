"""
fano_core_36.py
Complete characterisation of the 36 order-7 non-Weinberg elements of Z_62.

Questions:
1. What are they as matrices?
2. S_3-orbit structure?
3. Do they form a subgroup, coset, or union of conjugacy classes?
4. What do they generate?
5. Conjugacy class split: which orientation contributes how many?
6. Physical interpretation: what do they mediate?
"""
import sys
import numpy as np
from collections import Counter, defaultdict, deque
from itertools import product as iproduct

sys.path.insert(0, '.')
from psl27_core import (build_group, classify_strata, conjugacy_classes,
                         mat_key, mat_inv_f2)

print("=" * 72)
print("  FANO CORE 36: THE NON-WEINBERG ELEMENTS OF Z_62")
print("=" * 72)

elems, e2i, mul, inv_table, ords, ID = build_group()
B_31, Z_62, T_75, z3, z3sq, W26, S3 = classify_strata(
    elems, e2i, mul, inv_table, ords, ID)
conj_cls = conjugacy_classes(mul, inv_table)

# The 36
F36 = sorted(Z_62 - W26)
assert len(F36) == 36
assert all(ords[e] == 7 for e in F36)

# Charge conjugation
antidiag = np.array([[0,0,1],[0,1,0],[1,0,0]], dtype=int)
C_idx = e2i[mat_key(antidiag)]
fix_C = {i for i in range(168) if mul[C_idx, mul[i, inv_table[C_idx]]] == i}

# Fano infrastructure
fano_pts = []
fano_labels = []
for bits in iproduct([0,1], repeat=3):
    if any(b for b in bits):
        fano_pts.append(np.array(bits, dtype=int))
        fano_labels.append(''.join(map(str, bits)))

def fano_perm(elem_idx):
    M = elems[elem_idx]
    perm = []
    for fp in fano_pts:
        image = (M @ fp) % 2
        perm.append(fano_labels.index(''.join(map(str, image.tolist()))))
    return tuple(perm)

def fano_cycle_type(elem_idx):
    perm = fano_perm(elem_idx)
    visited = [False] * 7
    cycles = []
    for i in range(7):
        if visited[i]:
            continue
        length = 0
        j = i
        while not visited[j]:
            visited[j] = True
            j = perm[j]
            length += 1
        cycles.append(length)
    return tuple(sorted(cycles, reverse=True))

# =========================================================================
# 1. EXPLICIT MATRICES
# =========================================================================
print("\n" + "-" * 72)
print("  1. THE 36 ELEMENTS AS F_2 MATRICES")
print("-" * 72)

print(f"\n  {'Idx':>4} {'Order':>6} {'ConjCls':>8} {'Fano cycle':>12} {'Fix(C)':>7} {'Matrix (row-major)'}")
print(f"  {'-'*4} {'-'*6} {'-'*8} {'-'*12} {'-'*7} {'-'*30}")
for e in F36:
    cls_idx = next(ci for ci, cls in enumerate(conj_cls) if e in cls)
    ct = fano_cycle_type(e)
    in_fix = e in fix_C
    mat_str = str(elems[e].flatten().tolist())
    print(f"  {e:4d} {ords[e]:6d} {cls_idx:8d} {str(ct):>12} {str(in_fix):>7} {mat_str}")

# =========================================================================
# 2. S_3-ORBIT STRUCTURE
# =========================================================================
print("\n" + "-" * 72)
print("  2. S_3-ORBIT STRUCTURE OF THE 36")
print("-" * 72)

remaining = set(F36)
orbits_36 = []
while remaining:
    x = min(remaining)
    orbit = set()
    for g in S3:
        orbit.add(mul[g, mul[x, inv_table[g]]])
    orbits_36.append(sorted(orbit))
    remaining -= orbit
orbits_36.sort(key=lambda o: (len(o), o[0]))

print(f"\n  Number of S_3-orbits: {len(orbits_36)}")
print(f"  Orbit sizes: {[len(o) for o in orbits_36]}")
print(f"  Sum: {sum(len(o) for o in orbits_36)}")

for i, orb in enumerate(orbits_36):
    cls_indices = sorted(set(next(ci for ci, cls in enumerate(conj_cls) if e in cls) for e in orb))
    n_fix_c = sum(1 for e in orb if e in fix_C)
    print(f"  Orbit {i+1} (size {len(orb)}): conj classes {cls_indices}, Fix(C) = {n_fix_c}")

# =========================================================================
# 3. SUBGROUP / COSET / CONJUGACY CLASS TESTS
# =========================================================================
print("\n" + "-" * 72)
print("  3. ALGEBRAIC STRUCTURE OF THE 36")
print("-" * 72)

F36_set = set(F36)

# Test: closed under multiplication?
products_in = 0
products_out = 0
product_strata = Counter()
for a in F36:
    for b in F36:
        p = mul[a, b]
        if p in F36_set:
            products_in += 1
        else:
            products_out += 1
        if p in B_31:
            product_strata['B'] += 1
        elif p in Z_62:
            product_strata['Z'] += 1
        elif p in T_75:
            product_strata['T'] += 1

total_products = len(F36)**2
print(f"\n  Closure test: {products_in}/{total_products} products land in F36 "
      f"({products_in/total_products*100:.1f}%)")
print(f"  Products outside F36: {products_out}")
print(f"  IS A SUBGROUP? {products_in == total_products and ID in F36_set}")
print(f"  Product landing distribution: {dict(product_strata)}")

# Test: union of conjugacy classes?
print(f"\n  Conjugacy class membership:")
cls_membership = defaultdict(int)
for e in F36:
    cls_idx = next(ci for ci, cls in enumerate(conj_cls) if e in cls)
    cls_membership[cls_idx] += 1

for cls_idx, count in sorted(cls_membership.items()):
    cls = conj_cls[cls_idx]
    print(f"    Class {cls_idx} (size {len(cls)}, order {ords[cls[0]]}): "
          f"{count}/{len(cls)} in F36 ({count/len(cls)*100:.1f}%)")

is_union_of_classes = all(
    count == len(conj_cls[ci]) or count == 0
    for ci, count in cls_membership.items()
)
print(f"\n  Is union of FULL conjugacy classes? {is_union_of_classes}")

# Test: coset of a subgroup?
# If F36 = gH for some subgroup H and element g, then |H| = 36.
# But 36 does not divide 168 (168/36 = 4.666...), so F36 cannot be a coset.
print(f"\n  Coset test: |F36| = 36, |G| = 168, 168/36 = {168/36:.3f}")
print(f"  36 divides 168? {168 % 36 == 0}")
print(f"  IS A COSET? {'IMPOSSIBLE' if 168 % 36 != 0 else 'possible'}")

# =========================================================================
# 4. GENERATION TEST
# =========================================================================
print("\n" + "-" * 72)
print("  4. WHAT DO THE 36 GENERATE?")
print("-" * 72)

# BFS generation from F36
generated = {ID}
frontier = set(F36) | {ID}
while frontier - generated:
    generated.update(frontier)
    new = set()
    for x in frontier:
        for g in F36:
            new.add(mul[x, g])
            new.add(mul[g, x])
    frontier = new - generated

print(f"  |<F36>| = {len(generated)}")
print(f"  Generates full PSL(2,7)? {len(generated) == 168}")

# What about just a few of them?
# Try single element generation
for e in F36[:3]:
    cyc = {ID}
    cur = e
    while cur not in cyc or cur == ID:
        if cur == ID and len(cyc) > 1:
            break
        cyc.add(cur)
        cur = mul[cur, e]
    print(f"  <elem {e}> has order {len(cyc)} = {ords[e]}")

# =========================================================================
# 5. CONJUGACY CLASS SPLIT: ORIENTATIONS OF THE FANO 7-CYCLE
# =========================================================================
print("\n" + "-" * 72)
print("  5. CONJUGACY CLASS SPLIT: FANO 7-CYCLE ORIENTATIONS")
print("-" * 72)

# PSL(2,7) has exactly 2 conjugacy classes of order-7 elements
# (corresponding to the two generators of Z_7, related by inversion)
order7_all = [i for i in range(168) if ords[i] == 7]
print(f"\n  Total order-7 elements in PSL(2,7): {len(order7_all)}")

# Identify which conjugacy classes contain order-7 elements
order7_classes = []
for ci, cls in enumerate(conj_cls):
    if ords[cls[0]] == 7:
        order7_classes.append(ci)
        print(f"  Class {ci}: {len(cls)} elements, order {ords[cls[0]]}")

assert len(order7_classes) == 2, f"Expected 2 order-7 classes, got {len(order7_classes)}"

# How many from each class are in F36?
for ci in order7_classes:
    cls = conj_cls[ci]
    in_f36 = sum(1 for e in cls if e in F36_set)
    in_z62 = sum(1 for e in cls if e in Z_62)
    in_w26 = sum(1 for e in cls if e in W26)
    in_t75 = sum(1 for e in cls if e in T_75)
    in_b31 = sum(1 for e in cls if e in B_31)
    print(f"\n  Class {ci} ({len(cls)} elements):")
    print(f"    In F36 (non-Weinberg Z_62): {in_f36}/{len(cls)}")
    print(f"    In Z_62 total: {in_z62}/{len(cls)}")
    print(f"    In W26: {in_w26}/{len(cls)}")
    print(f"    In T_75: {in_t75}/{len(cls)}")
    print(f"    In B_31: {in_b31}/{len(cls)}")

# Orientation test: for each order-7 element, compute its Fano permutation
# and check if it's the same 7-cycle or inverse
print(f"\n  Fano permutation analysis of order-7 elements:")
print(f"  (Each order-7 element acts as a 7-cycle on the Fano plane)")

# Get representative permutations from each class
for ci in order7_classes:
    rep = conj_cls[ci][0]
    perm = fano_perm(rep)
    # Compute the cycle
    cycle = [0]
    cur = perm[0]
    while cur != 0:
        cycle.append(cur)
        cur = perm[cur]
    print(f"  Class {ci} representative (elem {rep}): cycle = {cycle}")

# Check: are the two classes related by inversion (e -> e^{-1})?
cls_a = set(conj_cls[order7_classes[0]])
cls_b = set(conj_cls[order7_classes[1]])
inversions_a_to_b = sum(1 for e in cls_a if inv_table[e] in cls_b)
print(f"\n  Class A inverses land in Class B: {inversions_a_to_b}/{len(cls_a)}")
print(f"  Classes are inverse-paired: {inversions_a_to_b == len(cls_a)}")

# =========================================================================
# 6. DISTANCES TO WEINBERG 26
# =========================================================================
print("\n" + "-" * 72)
print("  6. BFS DISTANCES: F36 -> WEINBERG 26")
print("-" * 72)

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

W26_set = set(W26)
dist_to_w26 = {}
for e in F36:
    dist_to_w26[e] = bfs_distance_to_set(e, W26_set, mul, gen_set)

dist_counter = Counter(dist_to_w26.values())
print(f"\n  Distance distribution (F36 -> W26):")
for d in sorted(dist_counter.keys()):
    print(f"    distance {d}: {dist_counter[d]} elements")

# By S_3-orbit
for i, orb in enumerate(orbits_36):
    orb_dists = [dist_to_w26[e] for e in orb]
    print(f"  Orbit {i+1}: mean dist to W26 = {np.mean(orb_dists):.3f}, dists = {dict(Counter(orb_dists))}")

# =========================================================================
# 7. PRODUCT STRUCTURE: F36 x W26
# =========================================================================
print("\n" + "-" * 72)
print("  7. PRODUCTS BETWEEN F36 AND W26")
print("-" * 72)

# Where do F36 * W26 products land?
fw_products = Counter()
for a in F36:
    for b in W26:
        p = mul[a, b]
        if p in B_31:
            fw_products['B'] += 1
        elif p in Z_62:
            fw_products['Z'] += 1
        elif p in T_75:
            fw_products['T'] += 1

total_fw = len(F36) * len(W26)
print(f"\n  F36 * W26 product landing ({total_fw} products):")
for s in ['B', 'Z', 'T']:
    print(f"    {s}: {fw_products[s]} ({fw_products[s]/total_fw*100:.1f}%)")

# Expected if uniform: B=31/168, Z=62/168, T=75/168
print(f"\n  Expected (uniform): B={31/168*100:.1f}%, Z={62/168*100:.1f}%, T={75/168*100:.1f}%")

# =========================================================================
# 8. Z_3 STRUCTURE OF F36
# =========================================================================
print("\n" + "-" * 72)
print("  8. Z_3 STRUCTURE: HOW F36 RELATES TO THE Z_3 GENERATOR")
print("-" * 72)

# Each F36 element came from Z_3-closure of B_31.
# For each f in F36: find which b in B_31 it came from (z3*b or z3^2*b)
print(f"\n  Z_3 origin of each F36 element:")
f36_from_z3 = 0
f36_from_z3sq = 0
for f in F36:
    # Check: is f = z3 * b for some b in B_31?
    for b in B_31:
        if mul[z3, b] == f:
            f36_from_z3 += 1
            break
        if mul[z3sq, b] == f:
            f36_from_z3sq += 1
            break

print(f"  From z3 * B_31: {f36_from_z3}")
print(f"  From z3^2 * B_31: {f36_from_z3sq}")
print(f"  Total: {f36_from_z3 + f36_from_z3sq}")

# Which B_31 elements produce F36 (vs W26) under Z_3 action?
b31_to_f36 = set()
b31_to_w26 = set()
for b in B_31:
    img1 = mul[z3, b]
    img2 = mul[z3sq, b]
    if img1 in F36_set or img2 in F36_set:
        b31_to_f36.add(b)
    if img1 in W26 or img2 in W26:
        b31_to_w26.add(b)

print(f"\n  B_31 elements whose Z_3-images include F36: {len(b31_to_f36)}")
print(f"  B_31 elements whose Z_3-images include W26: {len(b31_to_w26)}")
print(f"  Overlap: {len(b31_to_f36 & b31_to_w26)}")

# Orders of these B_31 sources
if b31_to_f36:
    f36_source_orders = Counter(int(ords[b]) for b in b31_to_f36)
    print(f"  F36 source orders: {dict(sorted(f36_source_orders.items()))}")
if b31_to_w26:
    w26_source_orders = Counter(int(ords[b]) for b in b31_to_w26)
    print(f"  W26 source orders: {dict(sorted(w26_source_orders.items()))}")

# =========================================================================
# 9. COMMUTATOR STRUCTURE
# =========================================================================
print("\n" + "-" * 72)
print("  9. COMMUTATORS [F36, W26]")
print("-" * 72)

# Commutator [a, b] = a * b * a^-1 * b^-1
comm_landing = Counter()
for a in F36[:12]:  # sample for speed
    for b in list(W26)[:13]:
        comm = mul[a, mul[b, mul[inv_table[a], inv_table[b]]]]
        if comm in B_31:
            comm_landing['B'] += 1
        elif comm in Z_62:
            comm_landing['Z'] += 1
        elif comm in T_75:
            comm_landing['T'] += 1

total_comm = sum(comm_landing.values())
print(f"\n  [F36, W26] commutator landing (sample, {total_comm} products):")
for s in ['B', 'Z', 'T']:
    if s in comm_landing:
        print(f"    {s}: {comm_landing[s]} ({comm_landing[s]/total_comm*100:.1f}%)")

# =========================================================================
# SUMMARY
# =========================================================================
print("\n" + "=" * 72)
print("  SUMMARY: THE 36 NON-WEINBERG ELEMENTS OF Z_62")
print("=" * 72)

print(f"""
  COUNT: 36 = |Z_62| - |W26| = 62 - 26
  ALL ORDER 7 (Fano 7-cycle elements)

  S_3-ORBITS: {len(orbits_36)} orbits, sizes {[len(o) for o in orbits_36]}

  ALGEBRAIC STRUCTURE:
    Subgroup? {products_in == total_products and ID in F36_set}
    Coset? IMPOSSIBLE (36 does not divide 168)
    Union of full conjugacy classes? {is_union_of_classes}
    Generates: {len(generated)}/168 of PSL(2,7)

  CONJUGACY CLASS SPLIT:
    PSL(2,7) has 2 classes of order-7 elements (24 each, inverse-paired)
    [see detailed split above]

  PHYSICAL INTERPRETATION:
    The 36 are the Fano core elements that sit in Z_62 but outside W26.
    They are order-7 = full Fano-plane rotations.
    The W26 are order-3 = partial rotations (Z_3 subgroup action).
    The 36 mediate the FULL Fano symmetry within the weak threshold.
""")
