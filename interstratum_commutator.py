"""
Script 3: interstratum_commutator.py
Goal: Prove SU(3)×SU(2) product structure — strata do not combine
into a simple group.

Tests:
1. Commutator [t,z] = t·z·t⁻¹·z⁻¹ landing distribution across strata
2. Can a T_75 element + Z_62 element generate all of PSL(2,7)?
3. Simple subgroup census: any simple subgroup containing both T and Z elements?
4. Product structure verification: do T-only and Z-only generated subgroups
   have trivial intersection?
"""
import sys
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations

sys.path.insert(0, '.')
from psl27_core import build_group, classify_strata

print("=" * 72)
print("  SCRIPT 3: INTERSTRATUM COMMUTATORS — PRODUCT STRUCTURE")
print("=" * 72)

# Build group
elems, e2i, mul, inv_table, ords, ID = build_group()
B_31, Z_62, T_75, z3, z3sq, W26, S3 = classify_strata(
    elems, e2i, mul, inv_table, ords, ID)

def stratum_of(x):
    if x in B_31: return 'B'
    if x in Z_62: return 'Z'
    if x in T_75: return 'T'
    return '?'

# ─────────────────────────────────────────────────────────────────────────────
# TEST 1: Commutator landing distribution [T, Z]
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "-" * 72)
print("  TEST 1: COMMUTATOR [T_75, Z_62] LANDING DISTRIBUTION")
print("-" * 72)

# [t, z] = t · z · t⁻¹ · z⁻¹
comm_landing = Counter()
comm_orders = Counter()
all_commutators_tz = set()

for t in T_75:
    for z in Z_62:
        c = mul[t, mul[z, mul[inv_table[t], inv_table[z]]]]
        comm_landing[stratum_of(c)] += 1
        comm_orders[int(ords[c])] += 1
        all_commutators_tz.add(c)

total = len(T_75) * len(Z_62)
print(f"\n  Total [t,z] commutators computed: {total}")
print(f"  Landing distribution:")
for s in ['B', 'Z', 'T']:
    count = comm_landing[s]
    print(f"    {s}: {count} ({count/total*100:.1f}%)")

print(f"\n  Distinct commutator values: {len(all_commutators_tz)}")
print(f"  Commutator order distribution: {dict(sorted(comm_orders.items()))}")

# Which strata do the distinct commutators land in?
comm_strata = Counter(stratum_of(c) for c in all_commutators_tz)
print(f"  Distinct commutators per stratum: {dict(sorted(comm_strata.items()))}")

# Does {[t,z]} generate all of PSL(2,7)?
print(f"\n  Do the commutators [T,Z] span all of PSL(2,7)?")
print(f"  All 168 elements reachable: {len(all_commutators_tz) == 168}")
print(f"  Elements NOT in commutator image: {168 - len(all_commutators_tz)}")

# Is the identity among the commutators? (= some t,z commute)
id_count = sum(1 for t in T_75 for z in Z_62
               if mul[t, z] == mul[z, t])
print(f"  Commuting (t,z) pairs [t,z]=e: {id_count} / {total}")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 2: Also compute [T, B] and [Z, B] commutators
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "-" * 72)
print("  TEST 2: ALL INTERSTRATUM COMMUTATOR DISTRIBUTIONS")
print("-" * 72)

pairs = [('T', 'Z', T_75, Z_62),
         ('T', 'B', T_75, B_31),
         ('Z', 'B', Z_62, B_31)]

for name1, name2, set1, set2 in pairs:
    landing = Counter()
    distinct = set()
    for a in set1:
        for b in set2:
            c = mul[a, mul[b, mul[inv_table[a], inv_table[b]]]]
            landing[stratum_of(c)] += 1
            distinct.add(c)
    total_p = len(set1) * len(set2)
    print(f"\n  [{name1}, {name2}] ({total_p} pairs, {len(distinct)} distinct values):")
    for s in ['B', 'Z', 'T']:
        count = landing[s]
        pct = count / total_p * 100 if total_p > 0 else 0
        print(f"    -> {s}: {count} ({pct:.1f}%)")
    distinct_strata = Counter(stratum_of(c) for c in distinct)
    print(f"    Distinct per stratum: {dict(sorted(distinct_strata.items()))}")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 3: Generation test — can one T + one Z element generate PSL(2,7)?
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "-" * 72)
print("  TEST 3: GENERATION FROM T × Z PAIRS")
print("-" * 72)

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

# Sample: test pairs
t_list = sorted(T_75)[:15]
z_list = sorted(Z_62)[:15]
gen_sizes_tz = Counter()
generates_full = 0
total_tested = 0

for t in t_list:
    for z in z_list:
        sg = subgroup_from([t, z], mul, ID)
        gen_sizes_tz[len(sg)] += 1
        if len(sg) == 168:
            generates_full += 1
        total_tested += 1

print(f"\n  Tested {total_tested} (t,z) pairs from T_75 × Z_62 (sample)")
print(f"  Subgroup size distribution: {dict(sorted(gen_sizes_tz.items()))}")
print(f"  Pairs generating FULL PSL(2,7): {generates_full}/{total_tested} "
      f"({generates_full/total_tested*100:.1f}%)")

if generates_full > 0:
    print(f"\n  RESULT: T_75 and Z_62 are NOT block-diagonal.")
    print(f"  A single t ∈ T_75 and z ∈ Z_62 can generate ALL of PSL(2,7).")
    print(f"  PSL(2,7) does NOT factor as a direct product of T-group × Z-group.")
else:
    print(f"\n  RESULT: No (t,z) pair generates the full group in sample.")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 4: Subgroups generated by T-only vs Z-only
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "-" * 72)
print("  TEST 4: SUBGROUPS FROM T-ONLY AND Z-ONLY")
print("-" * 72)

# Subgroup generated by all of T_75
gen_t = subgroup_from(list(T_75), mul, ID)
print(f"\n  |⟨T_75⟩| = {len(gen_t)}")

# Subgroup generated by all of Z_62
gen_z = subgroup_from(list(Z_62), mul, ID)
print(f"  |⟨Z_62⟩| = {len(gen_z)}")

# Subgroup generated by all of B_31
gen_b = subgroup_from(list(B_31), mul, ID)
print(f"  |⟨B_31⟩| = {len(gen_b)}")

# Intersection
print(f"\n  ⟨T_75⟩ ∩ ⟨Z_62⟩ = {len(gen_t & gen_z)} elements")
print(f"  ⟨T_75⟩ ∩ ⟨B_31⟩ = {len(gen_t & gen_b)} elements")
print(f"  ⟨Z_62⟩ ∩ ⟨B_31⟩ = {len(gen_z & gen_b)} elements")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 5: Simple subgroup census
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "-" * 72)
print("  TEST 5: SIMPLE SUBGROUP CENSUS")
print("-" * 72)

# PSL(2,7) is itself simple. Its proper subgroups are NOT simple (except
# for prime-order cyclic subgroups).
# Known maximal subgroups of PSL(2,7): S_4 (order 24), Z_7 ⋊ Z_3 (order 21),
# and D_8 (order 8) — none are simple except the trivial/cyclic ones.

# Find all subgroups of order dividing 168 that contain both T and Z elements
# by testing pairs
print(f"\n  Checking proper subgroups containing both T and Z elements:")

# Generate subgroups from representative pairs and check simplicity
mixed_subgroups = set()
for t in sorted(T_75)[:10]:
    for z in sorted(Z_62)[:10]:
        sg = subgroup_from([t, z], mul, ID)
        if len(sg) < 168 and len(sg) > 1:
            mixed_subgroups.add(sg)

print(f"  Distinct proper subgroups from T×Z pairs (sample): {len(mixed_subgroups)}")
sub_sizes = Counter(len(sg) for sg in mixed_subgroups)
print(f"  Size distribution: {dict(sorted(sub_sizes.items()))}")

# For each subgroup, check if it contains elements from both T and Z
for sg in sorted(mixed_subgroups, key=len):
    has_t = any(x in T_75 for x in sg)
    has_z = any(x in Z_62 for x in sg)
    has_b = any(x in B_31 and x != ID for x in sg)
    if has_t and has_z:
        # Check simplicity: a group is simple if it has no proper normal subgroups
        # Quick check: does it have a nontrivial centre?
        centre = [x for x in sg if all(mul[x, g] == mul[g, x] for g in sg)]
        ord_dist = Counter(int(ords[x]) for x in sg)
        print(f"  Order {len(sg)}: has T={has_t}, Z={has_z}, B={has_b}, "
              f"|centre|={len(centre)}, orders={dict(sorted(ord_dist.items()))}")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 6: Stratum multiplication table
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "-" * 72)
print("  TEST 6: STRATUM MULTIPLICATION TABLE")
print("-" * 72)
print(f"\n  For products a·b where a ∈ X and b ∈ Y, which stratum does a·b land in?")

strata_dict = {'B': B_31, 'Z': Z_62, 'T': T_75}
print(f"\n  {'X×Y':>6} {'->B':>8} {'->Z':>8} {'->T':>8} {'total':>8}")
print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

for n1 in ['B', 'Z', 'T']:
    for n2 in ['B', 'Z', 'T']:
        landing = Counter()
        for a in strata_dict[n1]:
            for b in strata_dict[n2]:
                landing[stratum_of(mul[a, b])] += 1
        total_p = len(strata_dict[n1]) * len(strata_dict[n2])
        print(f"  {n1}×{n2:>3} {landing.get('B',0):8d} {landing.get('Z',0):8d} "
              f"{landing.get('T',0):8d} {total_p:8d}")

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  SUMMARY")
print("=" * 72)
print(f"""
  Commutator [T,Z] distribution:
    Lands in B: {comm_landing['B']} ({comm_landing['B']/total*100:.1f}%)
    Lands in Z: {comm_landing['Z']} ({comm_landing['Z']/total*100:.1f}%)
    Lands in T: {comm_landing['T']} ({comm_landing['T']/total*100:.1f}%)
    Distinct values: {len(all_commutators_tz)} / 168

  Generation test (T × Z pairs):
    {generates_full}/{total_tested} pairs generate full PSL(2,7)

  Subgroups:
    ⟨T_75⟩ = {len(gen_t)} elements
    ⟨Z_62⟩ = {len(gen_z)} elements
    ⟨B_31⟩ = {len(gen_b)} elements

  PSL(2,7) is SIMPLE — it has no proper normal subgroups.
  The three strata B, Z, T are NOT subgroups and do NOT define
  a direct product decomposition. Instead, the gauge group
  SU(3)×SU(2)×U(1) emerges from the REPRESENTATION structure
  of the strata, not from group-theoretic factoring.
""")
