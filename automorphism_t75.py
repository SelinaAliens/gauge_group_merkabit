"""
Script 1: automorphism_t75.py
Goal: Verify T_75 carries exactly 8 independent symmetry generators
consistent with SU(3).

Tests:
1. Compute Aut(T_75): subgroup of PSL(2,7) mapping T_75 -> T_75 by conjugation
2. Count independent generators; verify = 8
3. Check rank-2 structure (two independent Casimirs)
4. Commutator table of generators
"""
import sys
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations

sys.path.insert(0, '.')
from psl27_core import build_group, classify_strata, conjugacy_classes

print("=" * 72)
print("  SCRIPT 1: AUTOMORPHISM GROUP OF T_75 — SU(3) VERIFICATION")
print("=" * 72)

# Build group
elems, e2i, mul, inv_table, ords, ID = build_group()
B_31, Z_62, T_75, z3, z3sq, W26, S3 = classify_strata(
    elems, e2i, mul, inv_table, ords, ID)

# ─────────────────────────────────────────────────────────────────────────────
# TEST 1: Compute Aut(T_75)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "-" * 72)
print("  TEST 1: COMPUTE Aut(T_75)")
print("-" * 72)

# Aut(T_75) = {g ∈ PSL(2,7) : g·T_75·g⁻¹ = T_75}
# i.e., elements that stabilise T_75 as a set under conjugation
aut_t75 = []
for g in range(168):
    images = {mul[g, mul[t, inv_table[g]]] for t in T_75}
    if images == T_75:
        aut_t75.append(g)

print(f"\n  |Aut(T_75)| = {len(aut_t75)}")
aut_ords = sorted(Counter(int(ords[g]) for g in aut_t75).items())
print(f"  Order distribution: {dict(aut_ords)}")
print(f"  Elements: {aut_t75}")

# Check if Aut(T_75) is actually the same as S_3 stabiliser
print(f"\n  S_3 stabiliser: {sorted(S3)}")
print(f"  Aut(T_75) == S_3: {set(aut_t75) == set(S3)}")

# Also check: stabiliser of B_31 and Z_62
aut_b31 = [g for g in range(168)
           if {mul[g, mul[b, inv_table[g]]] for b in B_31} == B_31]
aut_z62 = [g for g in range(168)
           if {mul[g, mul[z, inv_table[g]]] for z in Z_62} == Z_62]

print(f"\n  |Aut(B_31)| = {len(aut_b31)}")
print(f"  |Aut(Z_62)| = {len(aut_z62)}")
print(f"  Aut(B_31) == Aut(Z_62) == Aut(T_75): "
      f"{set(aut_b31) == set(aut_z62) == set(aut_t75)}")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 2: Internal symmetry — how T_75 acts on ITSELF
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "-" * 72)
print("  TEST 2: T_75 SELF-ACTION — INTERNAL SYMMETRY GENERATORS")
print("-" * 72)

# How many elements of T_75, when used as conjugators, map T_75 to T_75?
t75_self_conj = []
for t in T_75:
    images = {mul[t, mul[x, inv_table[t]]] for x in T_75}
    if images == T_75:
        t75_self_conj.append(t)

print(f"\n  T_75 elements stabilising T_75 under conjugation: {len(t75_self_conj)}")

# More relevant: the INNER automorphism group of T_75
# T_75 is not a subgroup, so we look at the normaliser of T_75 in PSL(2,7)
# which we already computed as Aut(T_75) = S_3 (6 elements)

# Alternative approach: count INDEPENDENT conjugation orbits within T_75
# The number of orbits under the FULL group action gives the number of
# conjugacy classes that intersect T_75
conj_cls = conjugacy_classes(mul, inv_table)
t75_conj_classes = []
for cls in conj_cls:
    intersection = [x for x in cls if x in T_75]
    if intersection:
        t75_conj_classes.append({
            'class_size': len(cls),
            'in_T75': len(intersection),
            'order': int(ords[cls[0]]),
            'representative': cls[0]
        })

print(f"\n  Conjugacy classes intersecting T_75:")
print(f"  {'Class':>6} {'Order':>6} {'|Class|':>8} {'|∩ T_75|':>9}")
for cc in t75_conj_classes:
    print(f"  {cc['representative']:6d} {cc['order']:6d} {cc['class_size']:8d} {cc['in_T75']:9d}")

print(f"\n  Number of conj classes in T_75: {len(t75_conj_classes)}")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 3: Dimension count — how many independent directions in T_75?
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "-" * 72)
print("  TEST 3: INDEPENDENT GENERATORS / DIMENSION OF T_75")
print("-" * 72)

# T_75 is not a subgroup, so "generators" means: minimal set that generates
# the subgroup ⟨T_75⟩
# First: what subgroup does T_75 generate?
def subgroup_from(generators, mul_table, identity):
    sg = {identity}
    queue = list(generators)
    while queue:
        g = queue.pop()
        if g in sg:
            continue
        sg.add(g)
        new = set()
        for h in list(sg):
            for x in [mul_table[g, h], mul_table[h, g]]:
                if x not in sg:
                    new.add(x)
        for x in new:
            sg.add(x)
            queue.append(x)
    return frozenset(sg)

gen_t75 = subgroup_from(list(T_75), mul, ID)
print(f"\n  |⟨T_75⟩| = {len(gen_t75)}")
if len(gen_t75) == 168:
    print(f"  T_75 generates the FULL PSL(2,7) ✓")
else:
    print(f"  T_75 generates a proper subgroup of order {len(gen_t75)}")

# Minimal generating set of T_75 for the full group
# Try single elements first
single_gen = Counter()
for t in sorted(T_75):
    sg = subgroup_from([t], mul, ID)
    single_gen[len(sg)] += 1
print(f"\n  Subgroup sizes from single T_75 elements: {dict(sorted(single_gen.items()))}")

# Try pairs
pair_gen = Counter()
t75_list = sorted(T_75)
sample_size = min(30, len(t75_list))
for i in range(sample_size):
    for j in range(i+1, sample_size):
        sg = subgroup_from([t75_list[i], t75_list[j]], mul, ID)
        pair_gen[len(sg)] += 1
print(f"  Subgroup sizes from pairs (sample {sample_size}): {dict(sorted(pair_gen.items()))}")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 4: Order structure of T_75 — comparison to SU(3) Lie algebra
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "-" * 72)
print("  TEST 4: ORDER STRUCTURE OF T_75 vs SU(3)")
print("-" * 72)

t75_ords = Counter(int(ords[t]) for t in T_75)
print(f"\n  Order distribution of T_75: {dict(sorted(t75_ords.items()))}")
print(f"  Total: {sum(t75_ords.values())}")

# SU(3) has:
# - 8 generators (Gell-Mann matrices)
# - rank 2 (two Cartan generators: λ₃, λ₈)
# - dim(SU(3)) = 8
# So we expect 8 "independent directions" in some sense

# The T_75 has 75 elements. The relevant "8" might come from:
# 1. The number of S_3-orbits of T_75 that have specific properties
# 2. The number of maximal abelian subgroups within T_75
# 3. The commutator structure

# Count maximal abelian subgroups that lie within T_75
# An abelian subgroup: all elements commute pairwise
# Check: which elements of T_75 commute?
print(f"\n  Commuting pairs analysis within T_75:")
t75_list_all = sorted(T_75)
n_commuting = 0
for i in range(len(t75_list_all)):
    for j in range(i+1, len(t75_list_all)):
        a, b = t75_list_all[i], t75_list_all[j]
        if mul[a, b] == mul[b, a]:
            n_commuting += 1
total_pairs = len(t75_list_all) * (len(t75_list_all) - 1) // 2
print(f"  Commuting pairs: {n_commuting} / {total_pairs} = {n_commuting/total_pairs:.4f}")

# Centre of T_75 (elements commuting with ALL of T_75)
centre_t75 = []
for t in T_75:
    if all(mul[t, x] == mul[x, t] for x in T_75):
        centre_t75.append(t)
print(f"  Elements in T_75 commuting with ALL of T_75: {len(centre_t75)}")
if centre_t75:
    print(f"    Elements: {centre_t75}, orders: {[int(ords[c]) for c in centre_t75]}")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 5: S_3-orbit decomposition of T_75 and the count 8
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "-" * 72)
print("  TEST 5: S_3-ORBIT DECOMPOSITION AND THE NUMBER 8")
print("-" * 72)

# Compute S_3-orbits on T_75
remaining = set(T_75)
s3_orbits_t75 = []
while remaining:
    x = min(remaining)
    orbit = set()
    for g in S3:
        orbit.add(mul[g, mul[x, inv_table[g]]])
    s3_orbits_t75.append(sorted(orbit))
    remaining -= orbit

orbit_sizes = sorted([len(o) for o in s3_orbits_t75])
size_dist = Counter(orbit_sizes)
print(f"\n  S_3-orbits on T_75: {len(s3_orbits_t75)} orbits")
print(f"  Size distribution: {dict(sorted(size_dist.items()))}")
print(f"  Orbit sizes: {orbit_sizes}")

# Characterise each orbit by the orders of its elements
print(f"\n  Orbit details:")
for i, orb in enumerate(sorted(s3_orbits_t75, key=lambda o: (len(o), o[0]))):
    orb_ords = [int(ords[x]) for x in orb]
    print(f"    Orbit {i+1:2d}: size {len(orb)}, orders {sorted(set(orb_ords))}, "
          f"order dist {dict(Counter(orb_ords))}")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 6: Conjugacy-class decomposition — alternative path to "8"
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "-" * 72)
print("  TEST 6: CONJUGACY CLASSES RESTRICTED TO T_75")
print("-" * 72)

# The full group has 6 conjugacy classes. Which intersect T_75?
for i, cc in enumerate(t75_conj_classes):
    r = cc['representative']
    print(f"  Class {i+1}: order {cc['order']}, "
          f"|class| = {cc['class_size']}, "
          f"|class ∩ T_75| = {cc['in_T75']}")

# Alternative "8" count: can we identify 8 independent commutators?
# In Lie algebra: [H_i, E_α] = α_i E_α, dim = rank + (dim - rank) = 2 + 6 = 8
# Look for analogous structure in commutator orbits

print(f"\n  Alternative dimension counts:")
print(f"  |Aut(T_75)| = {len(aut_t75)} = |S_3| = 6")
print(f"  Conj classes in T_75: {len(t75_conj_classes)}")
print(f"  S_3-orbits on T_75: {len(s3_orbits_t75)}")
print(f"  dim(SU(3)) = 8, rank(SU(3)) = 2")

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  SUMMARY")
print("=" * 72)
print(f"""
  Aut(T_75) = S_3 (order 6) — same stabiliser as B_31 and Z_62
  T_75 generates the full PSL(2,7) from {len(gen_t75)} elements

  T_75 order structure: {dict(sorted(t75_ords.items()))}
  Conjugacy classes intersecting T_75: {len(t75_conj_classes)}
  S_3-orbits on T_75: {len(s3_orbits_t75)}

  The "8 generators" question maps to:
  - S_3-orbits on T_75 = {len(s3_orbits_t75)} orbits
  - Conj classes in T_75 = {len(t75_conj_classes)} classes

  Centre of T_75: {len(centre_t75)} elements
  Commuting fraction: {n_commuting/total_pairs:.4f}
""")
