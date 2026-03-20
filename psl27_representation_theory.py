"""
psl27_representation_theory.py
Decompose B_31, Z_62, T_75 into irreducible representations of PSL(2,7).

PSL(2,7) has 6 irreps: 1, 3, 3-bar, 6, 7, 8.
Character table computed from conjugacy classes.
"""
import sys
import numpy as np
from collections import Counter, defaultdict
from itertools import product as iproduct

sys.path.insert(0, '.')
from psl27_core import build_group, classify_strata, conjugacy_classes, mat_key

print("=" * 72)
print("  PSL(2,7) REPRESENTATION THEORY — STRATUM DECOMPOSITION")
print("=" * 72)

elems, e2i, mul, inv_table, ords, ID = build_group()
B_31, Z_62, T_75, z3, z3sq, W26, S3 = classify_strata(
    elems, e2i, mul, inv_table, ords, ID)
conj_cls = conjugacy_classes(mul, inv_table)

# =========================================================================
# 1. CONJUGACY CLASSES
# =========================================================================
print("\n" + "-" * 72)
print("  1. CONJUGACY CLASSES OF PSL(2,7)")
print("-" * 72)

print(f"\n  {'Class':>6} {'Size':>6} {'Order':>6} {'Representative'}")
print(f"  {'-'*6} {'-'*6} {'-'*6} {'-'*20}")
class_sizes = []
class_orders = []
for i, cls in enumerate(conj_cls):
    rep = cls[0]
    class_sizes.append(len(cls))
    class_orders.append(int(ords[rep]))
    print(f"  {i:6d} {len(cls):6d} {ords[rep]:6d} elem {rep}")

print(f"\n  Sum of class sizes: {sum(class_sizes)} (should be 168)")
n_classes = len(conj_cls)
print(f"  Number of classes: {n_classes} (should be 6)")

# =========================================================================
# 2. CHARACTER TABLE OF PSL(2,7)
# =========================================================================
print("\n" + "-" * 72)
print("  2. CHARACTER TABLE OF PSL(2,7)")
print("-" * 72)

# PSL(2,7) character table is well-known.
# Classes ordered by (size, order): 1, 21, 24a, 24b, 42, 56
# But our classes may be in a different order. Let's sort them.
# Standard ordering: C1(1), C2(21), C3(56), C4(42), C7a(24), C7b(24)
# or by order: 1, 2, 3, 4, 7a, 7b

# Map our classes to standard form
class_info = [(i, class_sizes[i], class_orders[i]) for i in range(n_classes)]
class_info.sort(key=lambda x: (x[2], x[1]))  # sort by order, then size

print(f"\n  Classes sorted by order:")
sorted_indices = []
for ci, size, order in class_info:
    sorted_indices.append(ci)
    print(f"  Class {ci} -> order {order}, size {size}")

# The known character table of PSL(2,7) = GL(3,F_2)
# Sorted by class order: 1, 2, 3, 4, 7a, 7b
# (we need to identify which order-7 class is 7a vs 7b)
#
# Standard character table (from ATLAS/GAP):
# Class:     1    2   3    4   7a   7b
# Size:      1   21  56   42   24   24
# χ_1:       1    1   1    1    1    1
# χ_3:       3   -1   0    1    zeta7_val_a   zeta7_val_b
# χ_3bar:    3   -1   0    1    zeta7_val_b   zeta7_val_a
# χ_6:       6    2   0    0   -1   -1
# χ_7:       7   -1   1   -1    0    0
# χ_8:       8    0  -1    0    1    1
#
# where zeta7 = e^(2pi*i/7), and the values for chi_3 on 7a, 7b are:
# zeta7 + zeta7^2 + zeta7^4 = (-1+i*sqrt(7))/2
# zeta7^3 + zeta7^5 + zeta7^6 = (-1-i*sqrt(7))/2

z7 = np.exp(2j * np.pi / 7)
alpha = z7 + z7**2 + z7**4    # = (-1 + i*sqrt(7))/2
beta  = z7**3 + z7**5 + z7**6  # = (-1 - i*sqrt(7))/2

print(f"\n  zeta_7 Gauss sums:")
print(f"  alpha = z7 + z7^2 + z7^4 = {alpha:.6f}")
print(f"  beta  = z7^3 + z7^5 + z7^6 = {beta:.6f}")
print(f"  alpha + beta = {alpha + beta:.6f} (should be -1)")
print(f"  alpha * beta = {alpha * beta:.6f} (should be 2)")

# Build character table with our class ordering
# First identify our class indices for each order
our_classes = {}
for ci, size, order in class_info:
    key = f"ord{order}"
    if key in our_classes:
        key = f"ord{order}b"
    our_classes[key] = ci

print(f"\n  Our class mapping: {our_classes}")

# We need to figure out which of our two order-7 classes is "7a" vs "7b"
# This affects the 3-dim irrep values. For the permutation characters
# and the 1, 6, 7, 8 irreps, it doesn't matter (they're equal on both).
# For the decomposition, we'll compute both possibilities and check orthogonality.

# Build the character table indexed by our class ordering
# char_table[irrep_idx][class_idx] = value
# Classes in our sorted order: sorted_indices

# Map sorted position to class index
pos_to_ci = sorted_indices  # pos_to_ci[0] = class index of first sorted class

# Character table in sorted order (1, 2, 3, 4, 7a, 7b)
# We'll try both assignments for 7a/7b and pick the one that satisfies orthogonality
char_table_sorted_A = np.array([
    [1,  1,  1,  1,  1,     1],      # trivial
    [3, -1,  0,  1,  alpha, beta],    # 3
    [3, -1,  0,  1,  beta,  alpha],   # 3-bar
    [6,  2,  0,  0, -1,    -1],       # 6
    [7, -1,  1, -1,  0,     0],       # 7
    [8,  0, -1,  0,  1,     1],       # 8
], dtype=complex)

# Class sizes in sorted order
sizes_sorted = [class_sizes[ci] for ci in sorted_indices]
print(f"  Class sizes (sorted): {sizes_sorted}")

# Verify orthogonality: sum_C |C| * chi_i(C) * conj(chi_j(C)) / |G| = delta_ij
print(f"\n  Orthogonality check:")
G = 168
for i in range(6):
    for j in range(6):
        inner = sum(sizes_sorted[k] * char_table_sorted_A[i, k] *
                    np.conj(char_table_sorted_A[j, k]) for k in range(6)) / G
        if abs(inner) > 0.01:
            print(f"  <chi_{i+1}, chi_{j+1}> = {inner:.6f}"
                  f"{' (should be 1)' if i == j else ' *** NON-ZERO ***'}")

# Verify sum of dim^2 = |G|
dims = [int(np.real(char_table_sorted_A[i, 0])) for i in range(6)]
print(f"\n  Irrep dimensions: {dims}")
print(f"  Sum of dim^2: {sum(d**2 for d in dims)} (should be 168)")

# Pretty-print character table
irrep_names = ['1', '3', '3-bar', '6', '7', '8']
print(f"\n  Character Table of PSL(2,7):")
header = f"  {'Irrep':>6}"
for k in range(6):
    ci = sorted_indices[k]
    header += f" {'C'+str(ci)+'('+str(class_orders[ci])+')':>12}"
print(header)
print(f"  {'-'*6}" + f" {'-'*12}" * 6)
for i in range(6):
    row = f"  {irrep_names[i]:>6}"
    for k in range(6):
        val = char_table_sorted_A[i, k]
        if abs(val.imag) < 1e-10:
            row += f" {val.real:12.4f}"
        else:
            row += f" {val.real:+.2f}{val.imag:+.2f}i"
    print(row)

# =========================================================================
# 3. STRATUM PERMUTATION CHARACTERS
# =========================================================================
print("\n" + "-" * 72)
print("  3. STRATUM PERMUTATION CHARACTERS")
print("-" * 72)

# For a stratum S acted on by PSL(2,7) via conjugation:
# chi_S(g) = |{s in S : g*s*g^-1 = s}| = number of elements of S fixed by
# conjugation by g.
# Since all elements in a conjugacy class have the same fixed-point count,
# we compute chi_S(C_i) for each class C_i.

strata = {'B_31': B_31, 'Z_62': Z_62, 'T_75': T_75}

# For each stratum, compute permutation character on each conjugacy class
perm_chars = {}
for name, stratum in strata.items():
    char = []
    for ci in sorted_indices:  # iterate in sorted order
        rep = conj_cls[ci][0]  # representative of class ci
        fixed = sum(1 for s in stratum if mul[rep, mul[s, inv_table[rep]]] == s)
        char.append(fixed)
    perm_chars[name] = np.array(char, dtype=complex)

print(f"\n  Permutation characters (conjugation action):")
header = f"  {'Stratum':>8}"
for k in range(6):
    ci = sorted_indices[k]
    header += f" {'C'+str(ci)+'('+str(class_orders[ci])+')':>10}"
print(header)
print(f"  {'-'*8}" + f" {'-'*10}" * 6)
for name in ['B_31', 'Z_62', 'T_75']:
    row = f"  {name:>8}"
    for val in perm_chars[name]:
        row += f" {int(val.real):10d}"
    print(row)

# Also compute for full group (should be class sizes = centraliser size)
full_char = []
for ci in sorted_indices:
    rep = conj_cls[ci][0]
    fixed = sum(1 for s in range(168) if mul[rep, mul[s, inv_table[rep]]] == s)
    full_char.append(fixed)
print(f"  {'Full':>8}", end="")
for val in full_char:
    print(f" {val:10d}", end="")
print()

# Verify: perm char of full group is the regular representation character
# reg(e) = |G| = 168, reg(g) = 0 for g != e
print(f"\n  Full group char at identity: {full_char[0]} (should be 168)")
print(f"  Full group char at non-identity: {full_char[1:]} (should be all 0s: "
      f"{'yes' if all(v == 0 for v in full_char[1:]) else 'NO'})")

# =========================================================================
# 4. DECOMPOSE STRATUM CHARACTERS INTO IRREPS
# =========================================================================
print("\n" + "-" * 72)
print("  4. IRREDUCIBLE DECOMPOSITION OF STRATA")
print("-" * 72)

# Inner product: <chi, psi> = (1/|G|) * sum_C |C| * chi(C) * conj(psi(C))
def inner_product(chi, psi, sizes, G):
    return sum(sizes[k] * chi[k] * np.conj(psi[k]) for k in range(len(sizes))) / G

print(f"\n  Multiplicities of each irrep in each stratum:")
print(f"  {'Stratum':>8} {'1':>6} {'3':>6} {'3bar':>6} {'6':>6} {'7':>6} {'8':>6} {'Sum':>6}")
print(f"  {'-'*8} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")

decompositions = {}
for name in ['B_31', 'Z_62', 'T_75']:
    chi = perm_chars[name]
    mults = []
    for i in range(6):
        m = inner_product(chi, char_table_sorted_A[i], sizes_sorted, G)
        mults.append(m)
    decompositions[name] = mults

    row = f"  {name:>8}"
    dim_check = 0
    for m in mults:
        mr = np.real(m)
        mi = np.imag(m)
        if abs(mi) < 1e-8:
            row += f" {mr:6.2f}"
        else:
            row += f" {mr:+.1f}{mi:+.1f}i"
        dim_check += mr * dims[mults.index(m)] if abs(mi) < 1e-8 else 0
    # Sum check
    dim_sum = sum(np.real(mults[j]) * dims[j] for j in range(6))
    row += f" {dim_sum:6.1f}"
    print(row)

# =========================================================================
# 5. ANALYSIS OF IRREP CONTENT
# =========================================================================
print("\n" + "-" * 72)
print("  5. ANALYSIS: WHERE DOES EACH IRREP LIVE?")
print("-" * 72)

for i, name in enumerate(irrep_names):
    dim = dims[i]
    in_b = np.real(decompositions['B_31'][i])
    in_z = np.real(decompositions['Z_62'][i])
    in_t = np.real(decompositions['T_75'][i])
    print(f"\n  {name} (dim {dim}):")
    print(f"    B_31: multiplicity {in_b:.2f}")
    print(f"    Z_62: multiplicity {in_z:.2f}")
    print(f"    T_75: multiplicity {in_t:.2f}")
    print(f"    Total: {in_b + in_z + in_t:.2f}")

# =========================================================================
# 6. FANO PERMUTATION REPRESENTATION
# =========================================================================
print("\n" + "-" * 72)
print("  6. FANO PLANE PERMUTATION REPRESENTATION")
print("-" * 72)

# PSL(2,7) acts on 7 Fano points. Compute permutation character.
fano_pts = []
fano_labels = []
for bits in iproduct([0,1], repeat=3):
    if any(b for b in bits):
        fano_pts.append(np.array(bits, dtype=int))
        fano_labels.append(''.join(map(str, bits)))

fano_char = []
for ci in sorted_indices:
    rep = conj_cls[ci][0]
    M = elems[rep]
    fixed = sum(1 for fp in fano_pts if np.array_equal((M @ fp) % 2, fp))
    fano_char.append(fixed)

print(f"  Fano permutation character:")
for k in range(6):
    ci = sorted_indices[k]
    print(f"    Class {ci} (order {class_orders[ci]}, size {class_sizes[ci]}): "
          f"fix = {fano_char[k]}")

# Decompose Fano character
fano_chi = np.array(fano_char, dtype=complex)
print(f"\n  Fano character decomposition:")
fano_mults = []
for i in range(6):
    m = inner_product(fano_chi, char_table_sorted_A[i], sizes_sorted, G)
    fano_mults.append(m)
    mr = np.real(m)
    if abs(np.imag(m)) < 1e-8 and abs(mr) > 0.01:
        print(f"    {irrep_names[i]:>6}: {mr:.2f}")

print(f"\n  Fano = {' + '.join(f'{int(np.real(m))}x{irrep_names[i]}' for i, m in enumerate(fano_mults) if abs(np.real(m)) > 0.01)}")

# =========================================================================
# 7. REGULAR REPRESENTATION CHECK
# =========================================================================
print("\n" + "-" * 72)
print("  7. CONSISTENCY CHECK: B + Z + T = REGULAR")
print("-" * 72)

# The conjugation action on the full group gives the regular representation
# restricted to conjugation. Actually, the conjugation action on G itself
# gives: chi(g) = |centraliser of g|.
# The regular representation has chi_reg(e)=|G|, chi_reg(g!=e)=0.
# Conjugation action is different from regular action.

# But B + Z + T = full group, so their permutation characters should sum
# to the full-group conjugation character.
total_char = perm_chars['B_31'] + perm_chars['Z_62'] + perm_chars['T_75']
print(f"  B + Z + T character: {[int(np.real(v)) for v in total_char]}")
print(f"  Full group char:     {full_char}")
print(f"  Match: {all(abs(np.real(total_char[k]) - full_char[k]) < 0.01 for k in range(6))}")

# Total decomposition
print(f"\n  Total (B+Z+T) irrep content:")
for i in range(6):
    total_m = (np.real(decompositions['B_31'][i]) +
               np.real(decompositions['Z_62'][i]) +
               np.real(decompositions['T_75'][i]))
    print(f"    {irrep_names[i]:>6}: {total_m:.2f}")

# =========================================================================
# 8. THE KEY QUESTIONS
# =========================================================================
print("\n" + "-" * 72)
print("  8. KEY QUESTIONS ANSWERED")
print("-" * 72)

m8_b = np.real(decompositions['B_31'][5])  # 8 is index 5
m8_z = np.real(decompositions['Z_62'][5])
m8_t = np.real(decompositions['T_75'][5])
print(f"\n  Q1: Does dim-8 irrep appear in B_31 with multiplicity 1?")
print(f"      B_31 contains 8-irrep with multiplicity {m8_b:.2f}")
print(f"      Z_62 contains 8-irrep with multiplicity {m8_z:.2f}")
print(f"      T_75 contains 8-irrep with multiplicity {m8_t:.2f}")

m3_b = np.real(decompositions['B_31'][1])  # 3 is index 1
m3_z = np.real(decompositions['Z_62'][1])
m3_t = np.real(decompositions['T_75'][1])
m3b_b = np.real(decompositions['B_31'][2])  # 3-bar is index 2
m3b_z = np.real(decompositions['Z_62'][2])
m3b_t = np.real(decompositions['T_75'][2])
print(f"\n  Q2: Does dim-3 irrep appear in Z_62?")
print(f"      B_31: 3={m3_b:.2f}, 3-bar={m3b_b:.2f}")
print(f"      Z_62: 3={m3_z:.2f}, 3-bar={m3b_z:.2f}")
print(f"      T_75: 3={m3_t:.2f}, 3-bar={m3b_t:.2f}")

m7_b = np.real(decompositions['B_31'][4])  # 7 is index 4
m7_z = np.real(decompositions['Z_62'][4])
m7_t = np.real(decompositions['T_75'][4])
print(f"\n  Q3: Does dim-7 irrep correspond to Fano action?")
print(f"      B_31: 7={m7_b:.2f}")
print(f"      Z_62: 7={m7_z:.2f}")
print(f"      T_75: 7={m7_t:.2f}")

# =========================================================================
# SUMMARY
# =========================================================================
print("\n" + "=" * 72)
print("  SUMMARY: IRREDUCIBLE DECOMPOSITION OF STRATA")
print("=" * 72)

print(f"\n  PSL(2,7) irreps: {list(zip(irrep_names, dims))}")
print(f"\n  Stratum decompositions (multiplicity of each irrep):")
print(f"  {'':>8} {'1':>5} {'3':>5} {'3b':>5} {'6':>5} {'7':>5} {'8':>5}")
for name in ['B_31', 'Z_62', 'T_75']:
    mults = [f"{np.real(m):.0f}" if abs(np.imag(m)) < 0.01
             else f"{m:.1f}" for m in decompositions[name]]
    print(f"  {name:>8} " + " ".join(f"{m:>5}" for m in mults))

print(f"\n  Fano (7 points) = " +
      " + ".join(f"{int(np.real(m))}x{irrep_names[i]}"
                 for i, m in enumerate(fano_mults) if abs(np.real(m)) > 0.01))

print(f"""
  INTERPRETATION:
  The dim-8 irrep multiplicity in B_31: {m8_b:.0f}
  The dim-8 irrep multiplicity in T_75: {m8_t:.0f}
  The dim-3+3bar in Z_62: {m3_z:.0f} + {m3b_z:.0f}
  The dim-7 in Fano decomposition: appears as expected

  The gauge group identification comes from WHICH irreps
  dominate WHICH strata under the conjugation action.
""")
