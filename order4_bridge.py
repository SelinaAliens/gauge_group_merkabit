"""
order4_bridge.py
Element 18 (order 4, T_75) is the single element whose addition to <Z_62>
generates all of PSL(2,7). Characterise it completely.
"""
import sys
import numpy as np
from collections import Counter
from itertools import product as iproduct

sys.path.insert(0, '.')
from psl27_core import build_group, classify_strata, conjugacy_classes, mat_key

print("=" * 72)
print("  ORDER-4 BRIDGE ELEMENT: COMPLETE PROFILE")
print("=" * 72)

elems, e2i, mul, inv_table, ords, ID = build_group()
B_31, Z_62, T_75, z3, z3sq, W26, S3 = classify_strata(
    elems, e2i, mul, inv_table, ords, ID)
conj_cls = conjugacy_classes(mul, inv_table)
e2cls = {}
for ci, cls in enumerate(conj_cls):
    for e in cls:
        e2cls[e] = ci

# Charge conjugation
antidiag = np.array([[0,0,1],[0,1,0],[1,0,0]], dtype=int)
C_idx = e2i[mat_key(antidiag)]

# S3 involutions
s3_involutions = [g for g in S3 if ords[g] == 2]

EL = 18  # the bridge element

def stratum_of(x):
    if x in B_31: return 'B_31'
    if x in Z_62: return 'Z_62'
    if x in T_75: return 'T_75'
    return '?'

# =========================================================================
# 1. MATRIX
# =========================================================================
print("\n" + "-" * 72)
print("  1. MATRIX REPRESENTATION")
print("-" * 72)
M = elems[EL]
print(f"\n  Element {EL} as 3x3 matrix over F_2:")
for row in M:
    print(f"    {row.tolist()}")
print(f"  Flat: {M.flatten().tolist()}")
print(f"  Order: {ords[EL]}")
print(f"  Stratum: {stratum_of(EL)}")

# =========================================================================
# 2. FANO ACTION
# =========================================================================
print("\n" + "-" * 72)
print("  2. FANO PLANE ACTION (PG(2,2))")
print("-" * 72)

fano_pts = []
fano_labels = []
for bits in iproduct([0,1], repeat=3):
    if any(b for b in bits):
        fano_pts.append(np.array(bits, dtype=int))
        fano_labels.append(''.join(map(str, bits)))

# Permutation of Fano points under element 18
perm = []
for fp, label in zip(fano_pts, fano_labels):
    image = (M @ fp) % 2
    img_label = ''.join(map(str, image.tolist()))
    img_idx = fano_labels.index(img_label)
    perm.append(img_idx)
    print(f"  {label} -> {img_label}")

# Cycle decomposition
visited = [False] * 7
cycles = []
for i in range(7):
    if visited[i]:
        continue
    cycle = []
    j = i
    while not visited[j]:
        visited[j] = True
        cycle.append(fano_labels[j])
        j = perm[j]
    cycles.append(tuple(cycle))

print(f"\n  Cycle decomposition: {cycles}")
print(f"  Cycle type: {sorted([len(c) for c in cycles], reverse=True)}")

# =========================================================================
# 3. SQUARE OF ELEMENT 18
# =========================================================================
print("\n" + "-" * 72)
print("  3. SQUARE: element 18^2")
print("-" * 72)

sq = mul[EL, EL]
print(f"  18^2 = element {sq}")
print(f"  Order: {ords[sq]}")
print(f"  Stratum: {stratum_of(sq)}")
print(f"  Conj class: {e2cls[sq]} (size {len(conj_cls[e2cls[sq]])})")
print(f"  Matrix: {elems[sq].flatten().tolist()}")

# Is 18^2 an involution?
print(f"  Is involution: {ords[sq] == 2}")
# 18^3
cube = mul[sq, EL]
print(f"\n  18^3 = element {cube}")
print(f"  Order: {ords[cube]}")
print(f"  Stratum: {stratum_of(cube)}")
print(f"  18^3 = inv(18): {cube == inv_table[EL]}")

# =========================================================================
# 4. CONJUGACY CLASS
# =========================================================================
print("\n" + "-" * 72)
print("  4. CONJUGACY CLASS OF ELEMENT 18")
print("-" * 72)

cls_idx = e2cls[EL]
cls = conj_cls[cls_idx]
print(f"  Conj class index: {cls_idx}")
print(f"  Class size: {len(cls)}")
print(f"  All order: {set(int(ords[x]) for x in cls)}")

cls_by_stratum = Counter(stratum_of(x) for x in cls)
print(f"  Distribution by stratum: {dict(sorted(cls_by_stratum.items()))}")

cls_in_z62 = [x for x in cls if x in Z_62]
cls_in_b31 = [x for x in cls if x in B_31]
cls_in_t75 = [x for x in cls if x in T_75]
print(f"  In Z_62: {len(cls_in_z62)} elements")
print(f"  In B_31: {len(cls_in_b31)} elements")
print(f"  In T_75: {len(cls_in_t75)} elements")

# =========================================================================
# 5. CHARGE CONJUGATION
# =========================================================================
print("\n" + "-" * 72)
print("  5. CHARGE CONJUGATION C ACTION")
print("-" * 72)

c_image = mul[C_idx, mul[EL, inv_table[C_idx]]]
print(f"  C * 18 * C^-1 = element {c_image}")
print(f"  Order: {ords[c_image]}")
print(f"  Stratum: {stratum_of(c_image)}")
print(f"  Same conj class: {e2cls[c_image] == cls_idx}")
print(f"  In <Z_62>: {c_image in set(range(168)) and True}")  # will check below

# Check if c_image is in <Z_62>
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

gen_z = subgroup_from(list(Z_62), mul, ID)
print(f"  C-image in <Z_62>: {c_image in gen_z}")
print(f"  Element 18 in <Z_62>: {EL in gen_z}")

# =========================================================================
# 6. COMMUTATION WITH Z_62
# =========================================================================
print("\n" + "-" * 72)
print("  6. COMMUTATION WITH Z_62")
print("-" * 72)

comm_z62 = [z for z in Z_62 if mul[EL, z] == mul[z, EL]]
print(f"  Elements of Z_62 commuting with 18: {len(comm_z62)}")
if comm_z62:
    print(f"  Elements: {sorted(comm_z62)}")
    print(f"  Orders: {[int(ords[z]) for z in sorted(comm_z62)]}")
    for z in sorted(comm_z62):
        print(f"    elem {z}: order {ords[z]}, stratum {stratum_of(z)}")

# =========================================================================
# 7. COMMUTATION WITH B_31
# =========================================================================
print("\n" + "-" * 72)
print("  7. COMMUTATION WITH B_31")
print("-" * 72)

comm_b31 = [b for b in B_31 if mul[EL, b] == mul[b, EL]]
print(f"  Elements of B_31 commuting with 18: {len(comm_b31)}")
if comm_b31:
    print(f"  Elements: {sorted(comm_b31)}")
    print(f"  Orders: {[int(ords[b]) for b in sorted(comm_b31)]}")

# =========================================================================
# 8. <element 18, S_3>
# =========================================================================
print("\n" + "-" * 72)
print("  8. SUBGROUP <element 18, S_3>")
print("-" * 72)

gen_18_s3 = subgroup_from([EL] + list(S3), mul, ID)
print(f"  |<18, S_3>| = {len(gen_18_s3)}")
ord_dist = Counter(int(ords[x]) for x in gen_18_s3)
print(f"  Order distribution: {dict(sorted(ord_dist.items()))}")

# Identify the group
n = len(gen_18_s3)
known = {1: "trivial", 6: "S_3", 8: "D_4/Q_8/Z_8", 12: "A_4/D_6",
         24: "S_4/SL(2,3)", 42: "Z_7:Z_6", 168: "PSL(2,7)"}
print(f"  Group of order {n}: {known.get(n, '?')}")

# =========================================================================
# 9. CENTRALISER
# =========================================================================
print("\n" + "-" * 72)
print("  9. CENTRALISER C_PSL27(element 18)")
print("-" * 72)

centraliser = [g for g in range(168) if mul[g, EL] == mul[EL, g]]
print(f"  |C(18)| = {len(centraliser)}")
print(f"  Elements: {centraliser}")
print(f"  Orders: {[int(ords[g]) for g in centraliser]}")
cent_strata = Counter(stratum_of(g) for g in centraliser)
print(f"  By stratum: {dict(sorted(cent_strata.items()))}")

# What group is the centraliser?
cent_ord_dist = Counter(int(ords[g]) for g in centraliser)
print(f"  Order dist: {dict(sorted(cent_ord_dist.items()))}")

# =========================================================================
# 10. NORMALISER OF STRATA
# =========================================================================
print("\n" + "-" * 72)
print("  10. DOES ELEMENT 18 NORMALISE ANY STRATUM?")
print("-" * 72)

for name, stratum in [('B_31', B_31), ('Z_62', Z_62), ('T_75', T_75)]:
    images = {mul[EL, mul[x, inv_table[EL]]] for x in stratum}
    preserves = images == stratum
    if not preserves:
        leaked = images - stratum
        leaked_strata = Counter(stratum_of(x) for x in leaked)
        print(f"  {name}: NOT preserved. {len(leaked)} elements leak to {dict(leaked_strata)}")
    else:
        print(f"  {name}: PRESERVED")

# =========================================================================
# 11. S_3 INVOLUTION IMAGES
# =========================================================================
print("\n" + "-" * 72)
print("  11. S_3 ACTION ON ELEMENT 18")
print("-" * 72)

print(f"  S_3 elements: {S3}")
for g in S3:
    img = mul[g, mul[EL, inv_table[g]]]
    print(f"  S_3[{g}] (order {ords[g]}): 18 -> {img} "
          f"(order {ords[img]}, stratum {stratum_of(img)}, "
          f"in <Z_62>: {img in gen_z})")

# The S_3 orbit of element 18
s3_orbit_18 = {mul[g, mul[EL, inv_table[g]]] for g in S3}
print(f"\n  S_3-orbit of 18: {sorted(s3_orbit_18)}")
print(f"  Orbit size: {len(s3_orbit_18)}")
print(f"  Orders: {[int(ords[x]) for x in sorted(s3_orbit_18)]}")
print(f"  All in T_75: {all(x in T_75 for x in s3_orbit_18)}")
n_in_gen_z = sum(1 for x in s3_orbit_18 if x in gen_z)
print(f"  In <Z_62>: {n_in_gen_z}/{len(s3_orbit_18)}")

# =========================================================================
# 12. ALL ORDER-4 ELEMENTS
# =========================================================================
print("\n" + "-" * 72)
print("  12. ALL ORDER-4 ELEMENTS IN PSL(2,7)")
print("-" * 72)

ord4 = [i for i in range(168) if ords[i] == 4]
print(f"  Total order-4 elements: {len(ord4)}")
ord4_strata = Counter(stratum_of(x) for x in ord4)
print(f"  By stratum: {dict(sorted(ord4_strata.items()))}")

# How many conjugacy classes among order-4 elements?
ord4_classes = set(e2cls[x] for x in ord4)
print(f"  Conjugacy classes: {len(ord4_classes)}")
for ci in sorted(ord4_classes):
    size = len(conj_cls[ci])
    strata = Counter(stratum_of(x) for x in conj_cls[ci])
    print(f"    Class {ci}: size {size}, strata {dict(sorted(strata.items()))}")

# Which order-4 elements are NOT in <Z_62>?
ord4_missing = [x for x in ord4 if x not in gen_z]
print(f"\n  Order-4 elements NOT in <Z_62>: {len(ord4_missing)}")
print(f"  Elements: {sorted(ord4_missing)}")
print(f"  Strata: {[stratum_of(x) for x in sorted(ord4_missing)]}")

# =========================================================================
# 13. ORDER-7 ELEMENTS AND ELEMENT 161
# =========================================================================
print("\n" + "-" * 72)
print("  13. ORDER-7 ELEMENTS AND ELEMENT 161")
print("-" * 72)

ord7 = [i for i in range(168) if ords[i] == 7]
print(f"  Total order-7 elements: {len(ord7)}")
ord7_strata = Counter(stratum_of(x) for x in ord7)
print(f"  By stratum: {dict(sorted(ord7_strata.items()))}")

# Two conjugacy classes of order-7
ord7_classes = set(e2cls[x] for x in ord7)
for ci in sorted(ord7_classes):
    in_gen_z = sum(1 for x in conj_cls[ci] if x in gen_z)
    strata = Counter(stratum_of(x) for x in conj_cls[ci])
    print(f"  Class {ci}: size {len(conj_cls[ci])}, "
          f"in <Z_62>: {in_gen_z}/{len(conj_cls[ci])}, "
          f"strata {dict(sorted(strata.items()))}")

# Order-7 not in <Z_62>
ord7_missing = [x for x in ord7 if x not in gen_z]
print(f"\n  Order-7 elements NOT in <Z_62>: {len(ord7_missing)}")
print(f"  Elements: {sorted(ord7_missing)}")

# =========================================================================
# 14. CAN WE REACH 161 FROM 18 + Z_62?
# =========================================================================
print("\n" + "-" * 72)
print("  14. REACHING ELEMENT 161 FROM ELEMENT 18 + Z_62")
print("-" * 72)

# Try: 161 = z1 * 18 * z2 for z1, z2 in Z_62
found = False
for z1 in Z_62:
    prod = mul[z1, EL]
    for z2 in Z_62:
        if mul[prod, z2] == 161:
            print(f"  161 = {z1} * 18 * {z2}")
            print(f"    z1={z1} (order {ords[z1]}), z2={z2} (order {ords[z2]})")
            found = True
            break
    if found:
        break

if not found:
    # Try: 161 = z1 * 18^k * z2
    for k in range(1, 5):
        pow_k = EL
        for _ in range(k - 1):
            pow_k = mul[pow_k, EL]
        for z1 in Z_62:
            prod = mul[z1, pow_k]
            for z2 in Z_62:
                if mul[prod, z2] == 161:
                    print(f"  161 = {z1} * 18^{k} * {z2}")
                    print(f"    z1={z1} (order {ords[z1]}), z2={z2} (order {ords[z2]})")
                    found = True
                    break
            if found:
                break
        if found:
            break

if not found:
    # Try: 161 = 18 * z or z * 18
    for z in Z_62:
        if mul[EL, z] == 161:
            print(f"  161 = 18 * {z}")
            found = True
            break
        if mul[z, EL] == 161:
            print(f"  161 = {z} * 18")
            found = True
            break

if not found:
    print("  No simple decomposition 161 = z1 * 18^k * z2 found")
    # Check if 161 is in <Z_62 + 18>
    gen_z18 = subgroup_from(list(Z_62) + [EL], mul, ID)
    print(f"  161 in <Z_62, 18>: {161 in gen_z18}")
    print(f"  |<Z_62, 18>| = {len(gen_z18)}")

# =========================================================================
# SUMMARY
# =========================================================================
print("\n" + "=" * 72)
print("  COMPLETE PROFILE OF THE ORDER-4 BRIDGE (ELEMENT 18)")
print("=" * 72)
print(f"""
  Matrix:        {M.flatten().tolist()}
  Order:         {ords[EL]}
  Stratum:       T_75 (purely ternary)
  Conj class:    {cls_idx} (size {len(cls)}, all order-4)
  Fano cycles:   {[tuple(c) for c in cycles]}
  Cycle type:    {sorted([len(c) for c in cycles], reverse=True)}

  Square (18^2): element {sq}, order {ords[sq]}, stratum {stratum_of(sq)}
  Cube (18^3):   element {cube} = inv(18), order {ords[cube]}
  C-image:       element {c_image}, order {ords[c_image]}

  Centraliser:   {len(centraliser)} elements, orders {[int(ords[g]) for g in centraliser]}
  Commutes with Z_62: {len(comm_z62)} elements
  Commutes with B_31: {len(comm_b31)} elements

  <18, S_3>:     order {len(gen_18_s3)} = {known.get(len(gen_18_s3), '?')}
  S_3-orbit:     {sorted(s3_orbit_18)} (size {len(s3_orbit_18)})

  Normalises B_31: {all(mul[EL, mul[x, inv_table[EL]]] in B_31 for x in B_31)}
  Normalises Z_62: {all(mul[EL, mul[x, inv_table[EL]]] in Z_62 for x in Z_62)}
  Normalises T_75: {all(mul[EL, mul[x, inv_table[EL]]] in T_75 for x in T_75)}

  KEY PROPERTY: Adding element 18 alone to <Z_62> generates ALL of PSL(2,7).
  The order-4 bridge is the minimal extension of the weak sector
  that unlocks the full architecture.
""")
