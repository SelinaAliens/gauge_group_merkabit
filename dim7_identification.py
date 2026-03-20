"""
dim7_identification.py
The dim-7 irrep appears 3 times in Z_62 and 2 times in T_75.
What is it? How does it relate to SU(2)?
"""
import sys
import numpy as np
from collections import Counter, defaultdict
from itertools import product as iproduct, combinations

sys.path.insert(0, '.')
from psl27_core import build_group, classify_strata, conjugacy_classes, mat_key

print("=" * 72)
print("  DIM-7 IRREP IDENTIFICATION")
print("=" * 72)

elems, e2i, mul, inv_table, ords, ID = build_group()
B_31, Z_62, T_75, z3, z3sq, W26, S3 = classify_strata(
    elems, e2i, mul, inv_table, ords, ID)
conj_cls = conjugacy_classes(mul, inv_table)
e2cls = {e: ci for ci, cls in enumerate(conj_cls) for e in cls}

antidiag = np.array([[0,0,1],[0,1,0],[1,0,0]], dtype=int)
C_idx = e2i[mat_key(antidiag)]
s3_involutions = [g for g in S3 if ords[g] == 2]

# Fano points
fano_pts = []
fano_labels = []
for bits in iproduct([0,1], repeat=3):
    if any(b for b in bits):
        fano_pts.append(np.array(bits, dtype=int))
        fano_labels.append(''.join(map(str, bits)))

# Class ordering (sorted by order, then size)
class_info = [(i, len(conj_cls[i]), int(ords[conj_cls[i][0]])) for i in range(6)]
class_info.sort(key=lambda x: (x[2], x[1]))
sorted_indices = [ci for ci, _, _ in class_info]
sizes_sorted = [len(conj_cls[ci]) for ci in sorted_indices]
G = 168

# Character table
z7 = np.exp(2j * np.pi / 7)
alpha = z7 + z7**2 + z7**4
beta  = z7**3 + z7**5 + z7**6

char_table = np.array([
    [1,  1,  1,  1,  1,     1],
    [3, -1,  0,  1,  alpha, beta],
    [3, -1,  0,  1,  beta,  alpha],
    [6,  2,  0,  0, -1,    -1],
    [7, -1,  1, -1,  0,     0],
    [8,  0, -1,  0,  1,     1],
], dtype=complex)

irrep_names = ['1', '3', '3-bar', '6', '7', '8']
dims = [1, 3, 3, 6, 7, 8]

def inner_product(chi, psi):
    return sum(sizes_sorted[k] * chi[k] * np.conj(psi[k]) for k in range(6)) / G

def perm_char_of_action(action_func, set_size):
    """Compute permutation character of PSL(2,7) acting on a set.
    action_func(rep_matrix, item) -> image item
    Returns character indexed by sorted class order."""
    # We need to count fixed points for a representative of each class
    # But we need the set first
    pass

def decompose(chi):
    """Decompose a character into irreps."""
    mults = []
    for i in range(6):
        m = inner_product(chi, char_table[i])
        mults.append(m)
    return mults

# =========================================================================
# 1. VERIFY FANO = 1 + 6
# =========================================================================
print("\n" + "-" * 72)
print("  1. FANO POINT PERMUTATION = 1 + 6 (VERIFY)")
print("-" * 72)

fano_char = []
for ci in sorted_indices:
    rep = conj_cls[ci][0]
    M = elems[rep]
    fixed = sum(1 for fp in fano_pts if np.array_equal((M @ fp) % 2, fp))
    fano_char.append(fixed)
fano_chi = np.array(fano_char, dtype=complex)

fano_decomp = decompose(fano_chi)
print(f"  Fano point character: {[int(np.real(v)) for v in fano_chi]}")
print(f"  Decomposition:")
for i, m in enumerate(fano_decomp):
    if abs(np.real(m)) > 0.01:
        print(f"    {irrep_names[i]}: {np.real(m):.2f}")
print(f"  Fano = 1 + 6: {abs(np.real(fano_decomp[0]) - 1) < 0.01 and abs(np.real(fano_decomp[3]) - 1) < 0.01}")
print(f"  Dim-7 in Fano: {np.real(fano_decomp[4]):.4f} (should be 0)")

# =========================================================================
# 2. FANO LINE PERMUTATION
# =========================================================================
print("\n" + "-" * 72)
print("  2. FANO LINE PERMUTATION REPRESENTATION")
print("-" * 72)

# Fano lines: each line is a set of 3 collinear points in PG(2,2)
# A line in PG(2,2): {p, q, p+q} for distinct non-zero p, q in F_2^3
# There are 7 lines in PG(2,2)
fano_lines = []
for i in range(7):
    for j in range(i+1, 7):
        p, q = fano_pts[i], fano_pts[j]
        r = (p + q) % 2
        r_label = ''.join(map(str, r.tolist()))
        if r_label in fano_labels:
            k = fano_labels.index(r_label)
            line = frozenset([i, j, k])
            if line not in fano_lines:
                fano_lines.append(line)

print(f"  Number of Fano lines: {len(fano_lines)}")
for line in fano_lines:
    pts = sorted(line)
    print(f"    {{ {', '.join(fano_labels[p] for p in pts)} }}")

# Permutation character of line action
line_char = []
for ci in sorted_indices:
    rep = conj_cls[ci][0]
    M = elems[rep]
    # Compute permutation on Fano points
    perm = []
    for fp in fano_pts:
        image = (M @ fp) % 2
        perm.append(fano_labels.index(''.join(map(str, image.tolist()))))
    # Count fixed lines
    fixed_lines = 0
    for line in fano_lines:
        image_line = frozenset(perm[p] for p in line)
        if image_line == line:
            fixed_lines += 1
    line_char.append(fixed_lines)

line_chi = np.array(line_char, dtype=complex)
line_decomp = decompose(line_chi)
print(f"\n  Fano line character: {[int(np.real(v)) for v in line_chi]}")
print(f"  Decomposition:")
for i, m in enumerate(line_decomp):
    if abs(np.real(m)) > 0.01:
        print(f"    {irrep_names[i]}: {np.real(m):.2f}")

# =========================================================================
# 3. FLAG (POINT-LINE INCIDENCE) PERMUTATION
# =========================================================================
print("\n" + "-" * 72)
print("  3. FLAG (POINT-ON-LINE) PERMUTATION REPRESENTATION")
print("-" * 72)

# A flag = (point, line) where point is on line
# Each line has 3 points, each point is on 3 lines -> 21 flags
flags = []
for li, line in enumerate(fano_lines):
    for p in sorted(line):
        flags.append((p, li))
print(f"  Number of flags: {len(flags)}")

flag_char = []
for ci in sorted_indices:
    rep = conj_cls[ci][0]
    M = elems[rep]
    # Point permutation
    pt_perm = []
    for fp in fano_pts:
        image = (M @ fp) % 2
        pt_perm.append(fano_labels.index(''.join(map(str, image.tolist()))))
    # Line permutation
    line_perm = []
    for line in fano_lines:
        image_line = frozenset(pt_perm[p] for p in line)
        line_perm.append(fano_lines.index(image_line))
    # Count fixed flags
    fixed_flags = 0
    for (p, li) in flags:
        if pt_perm[p] == p and line_perm[li] == li:
            # Actually: flag (p, li) maps to (pt_perm[p], line_perm[li])
            # Fixed if both p and li are fixed
            # But actually: just check if image flag equals original
            if (pt_perm[p], line_perm[li]) == (p, li):
                fixed_flags += 1
    flag_char.append(fixed_flags)

flag_chi = np.array(flag_char, dtype=complex)
flag_decomp = decompose(flag_chi)
print(f"  Flag character: {[int(np.real(v)) for v in flag_chi]}")
print(f"  Decomposition:")
for i, m in enumerate(flag_decomp):
    if abs(np.real(m)) > 0.01:
        print(f"    {irrep_names[i]}: {np.real(m):.2f}")

# =========================================================================
# 4. PAIRS OF POINTS PERMUTATION
# =========================================================================
print("\n" + "-" * 72)
print("  4. PAIRS OF FANO POINTS (21 PAIRS)")
print("-" * 72)

pairs = list(combinations(range(7), 2))
print(f"  Number of pairs: {len(pairs)}")

pair_char = []
for ci in sorted_indices:
    rep = conj_cls[ci][0]
    M = elems[rep]
    pt_perm = []
    for fp in fano_pts:
        image = (M @ fp) % 2
        pt_perm.append(fano_labels.index(''.join(map(str, image.tolist()))))
    fixed_pairs = 0
    for (a, b) in pairs:
        img = tuple(sorted([pt_perm[a], pt_perm[b]]))
        if img == (a, b):
            fixed_pairs += 1
    pair_char.append(fixed_pairs)

pair_chi = np.array(pair_char, dtype=complex)
pair_decomp = decompose(pair_chi)
print(f"  Pair character: {[int(np.real(v)) for v in pair_chi]}")
print(f"  Decomposition:")
for i, m in enumerate(pair_decomp):
    if abs(np.real(m)) > 0.01:
        print(f"    {irrep_names[i]}: {np.real(m):.2f}")

# =========================================================================
# 5. ANTI-FLAGS (POINT NOT ON LINE)
# =========================================================================
print("\n" + "-" * 72)
print("  5. ANTI-FLAGS (POINT NOT ON LINE, 28 PAIRS)")
print("-" * 72)

anti_flags = []
for li, line in enumerate(fano_lines):
    for p in range(7):
        if p not in line:
            anti_flags.append((p, li))
print(f"  Number of anti-flags: {len(anti_flags)}")

aflag_char = []
for ci in sorted_indices:
    rep = conj_cls[ci][0]
    M = elems[rep]
    pt_perm = []
    for fp in fano_pts:
        image = (M @ fp) % 2
        pt_perm.append(fano_labels.index(''.join(map(str, image.tolist()))))
    line_perm = []
    for line in fano_lines:
        image_line = frozenset(pt_perm[p] for p in line)
        line_perm.append(fano_lines.index(image_line))
    fixed = sum(1 for (p, li) in anti_flags if pt_perm[p] == p and line_perm[li] == li)
    aflag_char.append(fixed)

aflag_chi = np.array(aflag_char, dtype=complex)
aflag_decomp = decompose(aflag_chi)
print(f"  Anti-flag character: {[int(np.real(v)) for v in aflag_chi]}")
print(f"  Decomposition:")
for i, m in enumerate(aflag_decomp):
    if abs(np.real(m)) > 0.01:
        print(f"    {irrep_names[i]}: {np.real(m):.2f}")

# =========================================================================
# 6. SCHUR INDICATOR OF DIM-7
# =========================================================================
print("\n" + "-" * 72)
print("  6. SCHUR INDICATOR OF DIM-7 IRREP")
print("-" * 72)

# Schur indicator: nu(chi) = (1/|G|) sum_g chi(g^2)
# = 1 means real (orthogonal), -1 means quaternionic (symplectic), 0 means complex
# For each conjugacy class C_i with rep g_i: g_i^2 lands in some class C_j
# chi_7(g^2) needs: for each class, find which class g^2 belongs to

for irrep_idx in range(6):
    nu = 0
    for ci in sorted_indices:
        rep = conj_cls[ci][0]
        sq = mul[rep, rep]
        sq_class_sorted_pos = next(k for k, cj in enumerate(sorted_indices)
                                    if sq in conj_cls[cj])
        nu += len(conj_cls[ci]) * char_table[irrep_idx, sq_class_sorted_pos]
    nu /= G
    indicator = "REAL (orthogonal)" if abs(np.real(nu) - 1) < 0.01 else \
                "QUATERNIONIC (symplectic)" if abs(np.real(nu) + 1) < 0.01 else \
                "COMPLEX (unitary)" if abs(nu) < 0.01 else f"??? ({nu})"
    print(f"  {irrep_names[irrep_idx]:>6} (dim {dims[irrep_idx]}): "
          f"nu = {np.real(nu):+.4f} -> {indicator}")

# =========================================================================
# 7. DIM-7 RESTRICTED TO S_3 SUBGROUP
# =========================================================================
print("\n" + "-" * 72)
print("  7. DIM-7 RESTRICTED TO S_3 SUBGROUP")
print("-" * 72)

# S_3 has 3 conjugacy classes: {e}(1), {involutions}(3), {order-3}(2)
# S_3 irreps: trivial(1), sign(1), standard(2)
# S_3 character table:
#   class:  e   inv  ord3
#   triv:   1    1    1
#   sign:   1   -1    1
#   std:    2    0   -1

# Restriction of chi_7 to S_3: evaluate chi_7 on S_3 elements
print(f"  S_3 elements: {S3}")
print(f"  Orders: {[int(ords[g]) for g in S3]}")

# Classify S_3 elements into S_3 conjugacy classes
s3_classes = defaultdict(list)
for g in S3:
    s3_classes[int(ords[g])].append(g)
print(f"  S_3 classes: {dict(s3_classes)}")

# For dim-7 character: evaluate on S_3 elements
# We need the actual character values, not just from the table
# chi_7 on class of order k: look up in our sorted table
chi_7 = char_table[4]  # dim-7 is index 4

print(f"\n  chi_7 values on S_3 elements:")
for g in S3:
    # Find which PSL(2,7) conjugacy class g belongs to
    g_class = e2cls[g]
    g_sorted_pos = sorted_indices.index(g_class)
    val = chi_7[g_sorted_pos]
    print(f"    elem {g} (order {ords[g]}): chi_7 = {np.real(val):.4f}")

# Restricted character on S_3 classes: (e: 7, inv: -1, ord3: 1)
# S_3 decomposition:
# <res, triv> = (1*7 + 3*(-1) + 2*1) / 6 = (7-3+2)/6 = 6/6 = 1
# <res, sign> = (1*7 + 3*(+1) + 2*1) / 6 -- wait, sign(inv) = -1
# <res, sign> = (1*7 + 3*(-1)*(-1) + 2*1*1) / 6 = (7+3+2)/6 = 12/6 = 2
# <res, std>  = (1*7*2 + 3*(-1)*0 + 2*1*(-1)) / 6 = (14+0-2)/6 = 12/6 = 2

# Let's compute properly
s3_char_table = np.array([
    [1,  1,  1],   # trivial
    [1, -1,  1],   # sign
    [2,  0, -1],   # standard
], dtype=float)
s3_sizes = [1, 3, 2]  # e, inv, ord3

# chi_7 restricted to S_3: values on (e, inv, ord3)
chi_7_on_e = 7
chi_7_on_inv = -1  # order-2 class has chi_7 = -1
chi_7_on_ord3 = 1  # order-3 class has chi_7 = 1
chi_7_res = np.array([chi_7_on_e, chi_7_on_inv, chi_7_on_ord3], dtype=float)

print(f"\n  chi_7 restricted to S_3: {chi_7_res}")
print(f"  S_3 class sizes: {s3_sizes}")

for i, name in enumerate(['trivial', 'sign', 'standard(2)']):
    m = sum(s3_sizes[k] * chi_7_res[k] * s3_char_table[i, k] for k in range(3)) / 6
    print(f"  <chi_7|_S3, {name}> = {m:.4f}")

print(f"\n  Dim-7 restricted to S_3 = 1*triv + 2*sign + 2*standard")
print(f"  Check: 1*1 + 2*1 + 2*2 = 1 + 2 + 4 = 7 ✓")
print(f"  Regular rep of S_3 = 1*triv + 1*sign + 2*standard (dim 6)")
print(f"  So: dim-7|_S3 = triv + sign + regular? "
      f"{1 + 2 + 4 == 7}")
print(f"  Actually: 1*triv + 2*sign + 2*std = triv + (sign + 2*std) = triv + ?")
print(f"  Regular = 1*triv + 1*sign + 2*std. So:")
print(f"  dim-7|_S3 = regular + sign = (1+1+2+2) + (1) -- NO: 6+1=7 but sign appears 2 not 1+1")
print(f"  dim-7|_S3 = 1*triv + 2*sign + 2*std (irreducible S_3 decomposition)")

# =========================================================================
# 8. ALL IRREPS RESTRICTED TO S_3
# =========================================================================
print("\n" + "-" * 72)
print("  8. ALL PSL(2,7) IRREPS RESTRICTED TO S_3")
print("-" * 72)

# For each PSL(2,7) irrep, compute restriction to S_3
# S_3 conjugacy classes within PSL(2,7):
# e (order 1) -> PSL class of order 1
# involutions (order 2) -> PSL class of order 2
# order-3 (order 3) -> PSL class of order 3

# Map S_3 class types to sorted PSL(2,7) class positions
# sorted_indices maps sorted position -> class index
# Position 0 = order 1, position 1 = order 2, position 2 = order 3
pos_e = 0
pos_inv = 1
pos_ord3 = 2

print(f"\n  {'PSL irrep':>10} {'on e':>6} {'on inv':>7} {'on o3':>6}  "
      f"{'triv':>5} {'sign':>5} {'std':>5}")
print(f"  {'-'*10} {'-'*6} {'-'*7} {'-'*6}  {'-'*5} {'-'*5} {'-'*5}")

for ir in range(6):
    chi_e = np.real(char_table[ir, pos_e])
    chi_inv = np.real(char_table[ir, pos_inv])
    chi_ord3 = np.real(char_table[ir, pos_ord3])
    chi_res = [chi_e, chi_inv, chi_ord3]

    m_triv = sum(s3_sizes[k] * chi_res[k] * s3_char_table[0, k] for k in range(3)) / 6
    m_sign = sum(s3_sizes[k] * chi_res[k] * s3_char_table[1, k] for k in range(3)) / 6
    m_std  = sum(s3_sizes[k] * chi_res[k] * s3_char_table[2, k] for k in range(3)) / 6

    print(f"  {irrep_names[ir]:>10} {chi_e:6.1f} {chi_inv:7.1f} {chi_ord3:6.1f}  "
          f"{m_triv:5.1f} {m_sign:5.1f} {m_std:5.1f}")

# =========================================================================
# 9. WHERE DOES DIM-7 APPEAR IN NATURAL ACTIONS?
# =========================================================================
print("\n" + "-" * 72)
print("  9. SUMMARY: DIM-7 IN NATURAL ACTIONS")
print("-" * 72)

actions = {
    'Fano points (7)': fano_decomp,
    'Fano lines (7)': line_decomp,
    'Flags (21)': flag_decomp,
    'Pairs (21)': pair_decomp,
    'Anti-flags (28)': aflag_decomp,
}

print(f"\n  {'Action':>20} {'dim-7 mult':>11}")
print(f"  {'-'*20} {'-'*11}")
for name, decomp in actions.items():
    m7 = np.real(decomp[4])
    print(f"  {name:>20} {m7:11.2f}")

# =========================================================================
# 10. WEINBERG 26 PERMUTATION CHARACTER
# =========================================================================
print("\n" + "-" * 72)
print("  10. WEINBERG 26 CONJUGATION CHARACTER")
print("-" * 72)

# Conjugation action on the 26 Weinberg elements
w26_char = []
for ci in sorted_indices:
    rep = conj_cls[ci][0]
    fixed = sum(1 for w in W26 if mul[rep, mul[w, inv_table[rep]]] == w)
    w26_char.append(fixed)

w26_chi = np.array(w26_char, dtype=complex)
w26_decomp = decompose(w26_chi)
print(f"  Weinberg 26 character: {[int(np.real(v)) for v in w26_chi]}")
print(f"  Decomposition:")
for i, m in enumerate(w26_decomp):
    mr = np.real(m)
    if abs(mr) > 0.01:
        print(f"    {irrep_names[i]}: {mr:.2f}")

m7_w26 = np.real(w26_decomp[4])
print(f"\n  Dim-7 in Weinberg 26: {m7_w26:.2f}")

# Also check: Z_62 minus Weinberg 26 (the 36 order-7 boundary elements)
z62_rest = Z_62 - W26
z62_rest_char = []
for ci in sorted_indices:
    rep = conj_cls[ci][0]
    fixed = sum(1 for z in z62_rest if mul[rep, mul[z, inv_table[rep]]] == z)
    z62_rest_char.append(fixed)

z62_rest_chi = np.array(z62_rest_char, dtype=complex)
z62_rest_decomp = decompose(z62_rest_chi)
print(f"\n  Z_62 \\ W_26 (36 elements) character: {[int(np.real(v)) for v in z62_rest_chi]}")
print(f"  Decomposition:")
for i, m in enumerate(z62_rest_decomp):
    mr = np.real(m)
    if abs(mr) > 0.01:
        print(f"    {irrep_names[i]}: {mr:.2f}")

# =========================================================================
# SUMMARY
# =========================================================================
print("\n" + "=" * 72)
print("  SUMMARY: DIM-7 IRREP IDENTIFICATION")
print("=" * 72)

print(f"""
  GEOMETRIC MEANING OF DIM-7:
  - Fano points = 1 + 6 (NOT dim-7)
  - Fano lines  = ? (check above)
  - Flags (21)  = contains dim-7? (check above)
  - Pairs (21)  = contains dim-7? (check above)
  - Anti-flags  = contains dim-7? (check above)

  SCHUR INDICATOR: (see above — real/orthogonal means SO, quaternionic means Sp)

  S_3 RESTRICTION:
  dim-7|_S3 = 1*triv + 2*sign + 2*std (verified)
  This is NOT triv + regular. It is its own decomposition.

  STRATUM DISTRIBUTION:
  dim-7 in B_31: ~0 (absent from matter)
  dim-7 in Z_62: 3 copies (dominant in weak sector)
  dim-7 in T_75: 2 copies

  WEINBERG SECTOR:
  dim-7 in W_26: {m7_w26:.2f}
  dim-7 in Z_62\\W_26: {np.real(z62_rest_decomp[4]):.2f}

  COMPLETE IRREP-STRATUM TABLE:
  Irrep  dim   B_31  Z_62  T_75
  1       1      2     3     1
  3       3      0     0     1
  3-bar   3      0     0     1
  6       6      2     0     3
  7       7      0     3     2
  8       8      1     4     4
""")
