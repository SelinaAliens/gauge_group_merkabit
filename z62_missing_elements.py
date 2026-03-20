"""
z62_missing_elements.py
The Z_62 boundary generates only 166/168 elements. Find the 2 missing.

Tests:
1. Identify the 2 missing elements
2. Their order, stratum, conjugacy class, Fix(C) membership
3. Do missing + Z_62 generate full PSL(2,7)?
4. Are they C-conjugate?
5. Are they the Z boson candidates from Paper 12?
"""
import sys
import numpy as np
from collections import Counter

sys.path.insert(0, '.')
from psl27_core import (build_group, classify_strata, conjugacy_classes,
                         mat_key)

print("=" * 72)
print("  Z_62 MISSING ELEMENTS — WHAT THE WEAK SECTOR CANNOT REACH")
print("=" * 72)

elems, e2i, mul, inv_table, ords, ID = build_group()
B_31, Z_62, T_75, z3, z3sq, W26, S3 = classify_strata(
    elems, e2i, mul, inv_table, ords, ID)
conj_cls = conjugacy_classes(mul, inv_table)
e2cls = {}
for ci, cls in enumerate(conj_cls):
    for e in cls:
        e2cls[e] = ci

# Charge conjugation C = antidiagonal
antidiag = np.array([[0,0,1],[0,1,0],[1,0,0]], dtype=int)
C_idx = e2i[mat_key(antidiag)]

# Fix(C)
fix_C = {i for i in range(168) if mul[C_idx, mul[i, inv_table[C_idx]]] == i}

# ── Generate <Z_62> ─────────────────────────────────────────────────────────
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

print("\n  Generating <Z_62>...")
gen_z = subgroup_from(list(Z_62), mul, ID)
print(f"  |<Z_62>| = {len(gen_z)}")

# ── Find missing elements ───────────────────────────────────────────────────
missing = set(range(168)) - gen_z
print(f"  Missing elements: {sorted(missing)}")

def stratum_of(x):
    if x in B_31: return 'B_31'
    if x in Z_62: return 'Z_62'
    if x in T_75: return 'T_75'
    return '?'

print("\n" + "-" * 72)
print("  PROPERTIES OF MISSING ELEMENTS")
print("-" * 72)

for m in sorted(missing):
    c_image = mul[C_idx, mul[m, inv_table[C_idx]]]
    print(f"\n  Element {m}:")
    print(f"    Matrix:       {elems[m].flatten().tolist()}")
    print(f"    Order:        {ords[m]}")
    print(f"    Stratum:      {stratum_of(m)}")
    print(f"    Conj class:   {e2cls[m]} (size {len(conj_cls[e2cls[m]])})")
    print(f"    In Fix(C):    {m in fix_C}")
    print(f"    C-image:      {c_image}")
    print(f"    Self-conj:    {c_image == m}")

# ── Are they C-conjugate to each other? ──────────────────────────────────────
print("\n" + "-" * 72)
print("  RELATION BETWEEN THE TWO MISSING ELEMENTS")
print("-" * 72)

m_list = sorted(missing)
if len(m_list) == 2:
    a, b = m_list
    c_of_a = mul[C_idx, mul[a, inv_table[C_idx]]]
    c_of_b = mul[C_idx, mul[b, inv_table[C_idx]]]
    print(f"\n  C({a}) = {c_of_a},  C({b}) = {c_of_b}")
    print(f"  C maps {a} <-> {b}: {c_of_a == b and c_of_b == a}")
    print(f"  Same conjugacy class: {e2cls[a] == e2cls[b]}")
    print(f"  Product a*b = {mul[a, b]} (order {ords[mul[a, b]]})")
    print(f"  Product b*a = {mul[b, a]} (order {ords[mul[b, a]]})")
    print(f"  Commute: {mul[a, b] == mul[b, a]}")

    # What subgroup do they generate together?
    gen_ab = subgroup_from([a, b], mul, ID)
    print(f"  |<{a}, {b}>| = {len(gen_ab)}")
    ord_dist = Counter(int(ords[x]) for x in gen_ab)
    print(f"  Order dist: {dict(sorted(ord_dist.items()))}")

# ── Do missing + Z_62 generate full group? ───────────────────────────────────
print("\n" + "-" * 72)
print("  GENERATION TEST: MISSING + Z_62")
print("-" * 72)

gen_full = subgroup_from(list(Z_62) + m_list, mul, ID)
print(f"  |<Z_62 + missing>| = {len(gen_full)}")
print(f"  Generates full PSL(2,7): {len(gen_full) == 168}")

# Test each missing element individually
for m in m_list:
    gen_one = subgroup_from(list(Z_62) + [m], mul, ID)
    print(f"  |<Z_62 + {m}>| = {len(gen_one)}")

# ── Paper 12 Z boson candidates ──────────────────────────────────────────────
print("\n" + "-" * 72)
print("  COMPARISON WITH PAPER 12 Z BOSON CANDIDATES")
print("-" * 72)

# Paper 12 found Fix(C) ∩ B_31 (non-identity) as Z boson candidates
z_candidates_p12 = {i for i in fix_C if i in B_31 and i != ID}
print(f"  Paper 12 Z boson candidates (Fix(C) ∩ B, non-id): {sorted(z_candidates_p12)}")
print(f"  Orders: {[int(ords[z]) for z in sorted(z_candidates_p12)]}")
print(f"  Missing elements: {m_list}")
print(f"  Overlap: {set(m_list) & z_candidates_p12}")

# What about Fix(C) ∩ T_75?
fix_C_in_T = {i for i in fix_C if i in T_75}
print(f"\n  Fix(C) ∩ T_75: {sorted(fix_C_in_T)} (orders {[int(ords[x]) for x in sorted(fix_C_in_T)]})")
print(f"  Overlap with missing: {set(m_list) & fix_C_in_T}")

# Full Fix(C) breakdown
print(f"\n  Complete Fix(C) = {sorted(fix_C)}")
print(f"  |Fix(C)| = {len(fix_C)}")
print(f"  Fix(C) by stratum:")
print(f"    B_31: {sorted(fix_C & B_31)}")
print(f"    Z_62: {sorted(fix_C & Z_62)}")
print(f"    T_75: {sorted(fix_C & T_75)}")

# ── What is special about <Z_62> = 166? ──────────────────────────────────────
print("\n" + "-" * 72)
print("  STRUCTURE OF THE 166-ELEMENT SUBGROUP")
print("-" * 72)

print(f"  Is <Z_62> a normal subgroup? (would make PSL(2,7)/<Z_62> well-defined)")
# Check: for all g in PSL(2,7), g * <Z_62> * g^-1 = <Z_62>?
is_normal = all(
    mul[g, mul[h, inv_table[g]]] in gen_z
    for g in range(168) for h in gen_z
)
print(f"  Normal: {is_normal}")

# If not normal, what is the normaliser?
if not is_normal:
    normaliser = [g for g in range(168)
                  if all(mul[g, mul[h, inv_table[g]]] in gen_z for h in gen_z)]
    print(f"  |N(<Z_62>)| = {len(normaliser)}")

# Order distribution of <Z_62>
gen_z_ords = Counter(int(ords[x]) for x in gen_z)
print(f"  Order distribution of <Z_62>: {dict(sorted(gen_z_ords.items()))}")

# Which elements of B_31 are in <Z_62>?
b_in_gen_z = B_31 & gen_z
b_missing = B_31 - gen_z
print(f"\n  B_31 ∩ <Z_62>: {len(b_in_gen_z)} / 31 elements")
print(f"  B_31 \\ <Z_62>: {sorted(b_missing)} (orders {[int(ords[x]) for x in sorted(b_missing)]})")

# Which elements of T_75 are in <Z_62>?
t_in_gen_z = T_75 & gen_z
t_missing = T_75 - gen_z
print(f"  T_75 ∩ <Z_62>: {len(t_in_gen_z)} / 75 elements")
print(f"  T_75 \\ <Z_62>: {sorted(t_missing)} (orders {[int(ords[x]) for x in sorted(t_missing)]})")

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  SUMMARY")
print("=" * 72)
print(f"""
  The Z_62 boundary generates 166/168 elements.
  Missing elements: {m_list}
  Orders: {[int(ords[m]) for m in m_list]}
  Strata: {[stratum_of(m) for m in m_list]}
  In Fix(C): {[m in fix_C for m in m_list]}
""")
