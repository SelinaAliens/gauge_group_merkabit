"""
b31_orbit_structure.py
B_31 has exactly 8 S_3-orbits. Is this the home of SU(3)?

Tests:
1. Enumerate all 8 orbits explicitly
2. Characterise each orbit (orders, Fano cycle types, Fix(C))
3. Do they decompose as 2 + 6 (Cartan + roots)?
4. C-pairing: 6 paired + 2 self-conjugate?
5. Rank-2 structure from S_3 involution fixed points
"""
import sys
import numpy as np
from collections import Counter, defaultdict
from itertools import product as iproduct

sys.path.insert(0, '.')
from psl27_core import build_group, classify_strata, conjugacy_classes, mat_key

print("=" * 72)
print("  B_31 ORBIT STRUCTURE: THE SU(3) IDENTIFICATION")
print("=" * 72)

elems, e2i, mul, inv_table, ords, ID = build_group()
B_31, Z_62, T_75, z3, z3sq, W26, S3 = classify_strata(
    elems, e2i, mul, inv_table, ords, ID)
conj_cls = conjugacy_classes(mul, inv_table)
e2cls = {e: ci for ci, cls in enumerate(conj_cls) for e in cls}

antidiag = np.array([[0,0,1],[0,1,0],[1,0,0]], dtype=int)
C_idx = e2i[mat_key(antidiag)]
fix_C = {i for i in range(168) if mul[C_idx, mul[i, inv_table[C_idx]]] == i}

s3_involutions = [g for g in S3 if ords[g] == 2]
s3_order3 = [g for g in S3 if ords[g] == 3]

def stratum_of(x):
    if x in B_31: return 'B'
    if x in Z_62: return 'Z'
    if x in T_75: return 'T'
    return '?'

# Fano infrastructure
fano_pts = []
fano_labels = []
for bits in iproduct([0,1], repeat=3):
    if any(b for b in bits):
        fano_pts.append(np.array(bits, dtype=int))
        fano_labels.append(''.join(map(str, bits)))

def fano_cycle_type(elem_idx):
    M = elems[elem_idx]
    perm = []
    for fp in fano_pts:
        image = (M @ fp) % 2
        perm.append(fano_labels.index(''.join(map(str, image.tolist()))))
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

def fano_fixed(elem_idx):
    M = elems[elem_idx]
    return [label for fp, label in zip(fano_pts, fano_labels)
            if np.array_equal((M @ fp) % 2, fp)]

# =========================================================================
# 1. ENUMERATE ALL 8 S_3-ORBITS OF B_31
# =========================================================================
print("\n" + "-" * 72)
print("  1. ALL 8 S_3-ORBITS OF B_31")
print("-" * 72)

remaining = set(B_31)
orbits_b31 = []
while remaining:
    x = min(remaining)
    orbit = set()
    for g in S3:
        orbit.add(mul[g, mul[x, inv_table[g]]])
    orbits_b31.append(sorted(orbit))
    remaining -= orbit

orbits_b31.sort(key=lambda o: (len(o), o[0]))

print(f"\n  {'Orb':>4} {'Size':>5} {'Orders':>12} {'Fano type':>14} "
      f"{'Fix(C)':>7} {'Elements'}")
print(f"  {'-'*4} {'-'*5} {'-'*12} {'-'*14} {'-'*7} {'-'*30}")

for i, orb in enumerate(orbits_b31):
    orb_ords = sorted(set(int(ords[x]) for x in orb))
    fano_types = sorted(set(fano_cycle_type(x) for x in orb))
    n_fix_c = sum(1 for x in orb if x in fix_C)
    print(f"  {i+1:4d} {len(orb):5d} {str(orb_ords):>12} {str(fano_types):>14} "
          f"{n_fix_c:7d} {orb}")

# =========================================================================
# 2. DETAILED ORBIT CHARACTERISATION
# =========================================================================
print("\n" + "-" * 72)
print("  2. DETAILED ORBIT CHARACTERISATION")
print("-" * 72)

for i, orb in enumerate(orbits_b31):
    print(f"\n  Orbit {i+1} (size {len(orb)}):")
    for e in orb:
        fps = fano_fixed(e)
        ct = fano_cycle_type(e)
        print(f"    elem {e:3d}: order {ords[e]}, Fano type {ct}, "
              f"fixed pts {fps}, Fix(C): {e in fix_C}, "
              f"conj class {e2cls[e]}")

# =========================================================================
# 3. C-PAIRING STRUCTURE
# =========================================================================
print("\n" + "-" * 72)
print("  3. CHARGE CONJUGATION PAIRING OF THE 8 ORBITS")
print("-" * 72)

# For each orbit, compute its C-image orbit
orbit_sets = [frozenset(orb) for orb in orbits_b31]

print(f"\n  C-image mapping of orbits:")
c_paired = []
self_conjugate = []
visited_orbs = set()

for i, orb in enumerate(orbits_b31):
    if i in visited_orbs:
        continue
    # C-image of this orbit
    c_orbit = frozenset(mul[C_idx, mul[e, inv_table[C_idx]]] for e in orb)

    # Find which orbit this maps to
    target = None
    for j, orb_j in enumerate(orbits_b31):
        if frozenset(orb_j) == c_orbit:
            target = j
            break

    if target == i:
        self_conjugate.append(i)
        visited_orbs.add(i)
        print(f"  Orbit {i+1} -> Orbit {i+1} (SELF-CONJUGATE)")
    else:
        c_paired.append((i, target))
        visited_orbs.add(i)
        visited_orbs.add(target)
        print(f"  Orbit {i+1} <-> Orbit {target+1} (PAIRED)")

print(f"\n  Self-conjugate orbits: {len(self_conjugate)} "
      f"(orbits {[i+1 for i in self_conjugate]})")
print(f"  Paired orbits: {len(c_paired)} pairs "
      f"({[f'{i+1}<->{j+1}' for i,j in c_paired]})")
print(f"  Decomposition: {len(self_conjugate)} + 2x{len(c_paired)} = "
      f"{len(self_conjugate)} + {2*len(c_paired)} = "
      f"{len(self_conjugate) + 2*len(c_paired)}")

has_6_plus_2 = (len(self_conjugate) == 2 and len(c_paired) == 3)
print(f"\n  Is 8 = 2 (self-conj) + 6 (3 pairs)? {has_6_plus_2}")

# =========================================================================
# 4. Fix(C) ∩ B_31
# =========================================================================
print("\n" + "-" * 72)
print("  4. Fix(C) ∩ B_31")
print("-" * 72)

fix_c_b31 = sorted(fix_C & B_31)
print(f"  |Fix(C) ∩ B_31| = {len(fix_c_b31)}")
print(f"  Elements: {fix_c_b31}")
for e in fix_c_b31:
    orb_idx = next(i for i, orb in enumerate(orbits_b31) if e in orb)
    print(f"    elem {e:3d}: order {ords[e]}, orbit {orb_idx+1} (size {len(orbits_b31[orb_idx])}), "
          f"Fano fixed {fano_fixed(e)}")

# Which orbits contain Fix(C) elements?
orbits_with_fix_c = set()
for e in fix_c_b31:
    for i, orb in enumerate(orbits_b31):
        if e in orb:
            orbits_with_fix_c.add(i)
print(f"\n  Orbits containing Fix(C) elements: {sorted(i+1 for i in orbits_with_fix_c)}")
print(f"  These are the self-conjugate orbits: "
      f"{sorted(i+1 for i in orbits_with_fix_c) == sorted(i+1 for i in self_conjugate)}")

# =========================================================================
# 5. RANK-2 STRUCTURE: S_3 INVOLUTION FIXED POINTS
# =========================================================================
print("\n" + "-" * 72)
print("  5. RANK-2 STRUCTURE FROM S_3 INVOLUTION FIXED POINTS")
print("-" * 72)

# For each S_3 involution, which B_31 elements are fixed?
print(f"\n  S_3 involutions: {s3_involutions} (orders {[int(ords[g]) for g in s3_involutions]})")

for inv_elem in s3_involutions:
    fixed = [b for b in B_31 if mul[inv_elem, mul[b, inv_table[inv_elem]]] == b]
    fixed_orbits = set()
    for e in fixed:
        for i, orb in enumerate(orbits_b31):
            if e in orb:
                fixed_orbits.add(i)
    print(f"\n  Involution {inv_elem} fixes {len(fixed)} B_31 elements: {sorted(fixed)}")
    print(f"    Orders: {[int(ords[e]) for e in sorted(fixed)]}")
    print(f"    In orbits: {sorted(i+1 for i in fixed_orbits)}")

    # Which orbits are POINTWISE fixed (all elements of orbit are fixed)?
    pointwise_fixed = [i for i in fixed_orbits
                       if all(e in fixed for e in orbits_b31[i])]
    print(f"    Orbits pointwise fixed: {[i+1 for i in pointwise_fixed]}")

# Are there exactly 2 orbits pointwise fixed by some involution?
# Check all involutions
print(f"\n  Pointwise-fixed orbit counts per involution:")
for inv_elem in s3_involutions:
    fixed = {b for b in B_31 if mul[inv_elem, mul[b, inv_table[inv_elem]]] == b}
    pw_fixed = [i for i, orb in enumerate(orbits_b31)
                if all(e in fixed for e in orb)]
    print(f"    Inv {inv_elem}: {len(pw_fixed)} pointwise-fixed orbits "
          f"({[i+1 for i in pw_fixed]})")

# =========================================================================
# 6. ORBIT SIZE STRUCTURE VS GLUON REPRESENTATION
# =========================================================================
print("\n" + "-" * 72)
print("  6. ORBIT SIZES AND GLUON REPRESENTATION")
print("-" * 72)

size_dist = Counter(len(orb) for orb in orbits_b31)
print(f"\n  Orbit size distribution: {dict(sorted(size_dist.items()))}")
print(f"  Sizes: {[len(orb) for orb in orbits_b31]}")

# SU(3) adjoint = 8: decomposes under Z_2 (charge conj) as 2 + 3 + 3-bar
# The 2 are the Cartan generators (self-conjugate)
# The 3 + 3-bar are the raising/lowering operators (paired)
# So: 2 self-conjugate orbits + 3 paired orbits = 2 + 6 = 8
# Test: do the self-conjugate orbits have different sizes from paired?

print(f"\n  Self-conjugate orbit sizes: "
      f"{[len(orbits_b31[i]) for i in self_conjugate]}")
print(f"  Paired orbit sizes (each pair):")
for i, j in c_paired:
    print(f"    Orbit {i+1} (size {len(orbits_b31[i])}) <-> "
          f"Orbit {j+1} (size {len(orbits_b31[j])})")

# Total element counts
self_conj_total = sum(len(orbits_b31[i]) for i in self_conjugate)
paired_total = sum(len(orbits_b31[i]) + len(orbits_b31[j]) for i,j in c_paired)
print(f"\n  Elements in self-conjugate orbits: {self_conj_total}")
print(f"  Elements in paired orbits: {paired_total}")
print(f"  Total: {self_conj_total + paired_total}")

# =========================================================================
# 7. ORDER-3 S_3 ELEMENTS: Z_3 ACTION ON ORBITS
# =========================================================================
print("\n" + "-" * 72)
print("  7. Z_3 ACTION ON THE 8 ORBITS")
print("-" * 72)

# The order-3 elements of S_3 act on the orbits by permutation
# Since each orbit is an S_3-orbit, the Z_3 subgroup acts within each orbit
# But we can check: does Z_3 permute some orbits?
print(f"\n  Z_3 elements of S_3: {s3_order3}")

for z3_elem in s3_order3:
    print(f"\n  Z_3 element {z3_elem} (order {ords[z3_elem]}):")
    for i, orb in enumerate(orbits_b31):
        # Image of orbit under this element
        image = frozenset(mul[z3_elem, mul[e, inv_table[z3_elem]]] for e in orb)
        target = next(j for j, orb_j in enumerate(orbits_b31) if frozenset(orb_j) == image)
        if target != i:
            print(f"    Orbit {i+1} -> Orbit {target+1}")
        else:
            print(f"    Orbit {i+1} -> Orbit {i+1} (fixed)")

# =========================================================================
# 8. T_75 ORBITS: SUB-STRUCTURE
# =========================================================================
print("\n" + "-" * 72)
print("  8. T_75 S_3-ORBITS (14 ORBITS) — ANY 8-DIMENSIONAL SUB-STRUCTURE?")
print("-" * 72)

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

print(f"\n  {'Orb':>4} {'Size':>5} {'Orders':>15} {'Fix(C)':>7}")
print(f"  {'-'*4} {'-'*5} {'-'*15} {'-'*7}")
for i, orb in enumerate(orbits_t75):
    orb_ords = sorted(set(int(ords[x]) for x in orb))
    n_fix_c = sum(1 for x in orb if x in fix_C)
    print(f"  {i+1:4d} {len(orb):5d} {str(orb_ords):>15} {n_fix_c:7d}")

# C-pairing of T_75 orbits
t75_orbit_sets = [frozenset(orb) for orb in orbits_t75]
t75_self_conj = []
t75_paired = []
visited_t = set()
for i, orb in enumerate(orbits_t75):
    if i in visited_t:
        continue
    c_orb = frozenset(mul[C_idx, mul[e, inv_table[C_idx]]] for e in orb)
    target = next(j for j, orb_j in enumerate(orbits_t75) if frozenset(orb_j) == c_orb)
    if target == i:
        t75_self_conj.append(i)
        visited_t.add(i)
    else:
        t75_paired.append((i, target))
        visited_t.add(i)
        visited_t.add(target)

print(f"\n  T_75 C-pairing: {len(t75_self_conj)} self-conjugate + "
      f"{len(t75_paired)} pairs = {len(t75_self_conj) + 2*len(t75_paired)}")
print(f"  Self-conjugate T_75 orbits: {[i+1 for i in t75_self_conj]}")

# Z_62 orbits too
remaining = set(Z_62)
orbits_z62 = []
while remaining:
    x = min(remaining)
    orbit = set()
    for g in S3:
        orbit.add(mul[g, mul[x, inv_table[g]]])
    orbits_z62.append(sorted(orbit))
    remaining -= orbit
orbits_z62.sort(key=lambda o: (len(o), o[0]))

z62_self_conj = []
z62_paired = []
visited_z = set()
for i, orb in enumerate(orbits_z62):
    if i in visited_z:
        continue
    c_orb = frozenset(mul[C_idx, mul[e, inv_table[C_idx]]] for e in orb)
    target = next(j for j, orb_j in enumerate(orbits_z62) if frozenset(orb_j) == c_orb)
    if target == i:
        z62_self_conj.append(i)
        visited_z.add(i)
    else:
        z62_paired.append((i, target))
        visited_z.add(i)
        visited_z.add(target)

print(f"\n  Z_62 C-pairing: {len(z62_self_conj)} self-conj + "
      f"{len(z62_paired)} pairs = {len(z62_self_conj) + 2*len(z62_paired)}")

# =========================================================================
# SUMMARY
# =========================================================================
print("\n" + "=" * 72)
print("  SUMMARY: B_31 ORBIT STRUCTURE")
print("=" * 72)

print(f"""
  B_31 has 8 S_3-orbits, sizes: {[len(o) for o in orbits_b31]}

  C-pairing decomposition:
    Self-conjugate: {len(self_conjugate)} orbits ({[i+1 for i in self_conjugate]})
    C-paired: {len(c_paired)} pairs ({[f'{i+1}<->{j+1}' for i,j in c_paired]})
    Structure: {len(self_conjugate)} + 2x{len(c_paired)} = {len(self_conjugate) + 2*len(c_paired)}
    Target (SU(3) adjoint): 2 + 2x3 = 8
    MATCH: {has_6_plus_2}

  Fix(C) ∩ B_31: {len(fix_c_b31)} elements in orbits {sorted(i+1 for i in orbits_with_fix_c)}
  Self-conjugate = Fix(C)-containing: {sorted(i+1 for i in orbits_with_fix_c) == sorted(i+1 for i in self_conjugate)}

  Comparison across strata:
    B_31: 8 orbits = {len(self_conjugate)} self-conj + {len(c_paired)} pairs
    Z_62: 11 orbits = {len(z62_self_conj)} self-conj + {len(z62_paired)} pairs
    T_75: 14 orbits = {len(t75_self_conj)} self-conj + {len(t75_paired)} pairs
""")
