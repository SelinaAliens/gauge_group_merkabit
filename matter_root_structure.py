"""
matter_root_structure.py
Test G_2 root structure of 12 order-4 flag elements and U(3) structure
of 9 line involutions in B_31.

Conjecture: The 12 order-4 flags are G_2 roots, the 9 order-2 line
involutions are U(3) generators (8 gluons + 1 photon), all in matter.
"""
import sys
import numpy as np
from collections import Counter, defaultdict, deque
from itertools import product as iproduct, combinations

sys.path.insert(0, '.')
from psl27_core import build_group, classify_strata, conjugacy_classes, mat_key

print("=" * 72)
print("  MATTER ROOT STRUCTURE: G_2 + U(3) IN B_31")
print("=" * 72)

elems, e2i, mul, inv_table, ords, ID = build_group()
B_31, Z_62, T_75, z3, z3sq, W26, S3 = classify_strata(
    elems, e2i, mul, inv_table, ords, ID)
conj_cls = conjugacy_classes(mul, inv_table)

# Charge conjugation
antidiag = np.array([[0,0,1],[0,1,0],[1,0,0]], dtype=int)
C_idx = e2i[mat_key(antidiag)]
fix_C = {i for i in range(168) if mul[C_idx, mul[i, inv_table[C_idx]]] == i}

# S_3 elements
s3_involutions = [g for g in S3 if ords[g] == 2]
s3_order3 = [g for g in S3 if ords[g] == 3]

# Fano infrastructure
fano_pts = []
fano_labels = []
for bits in iproduct([0,1], repeat=3):
    if any(b for b in bits):
        fano_pts.append(np.array(bits, dtype=int))
        fano_labels.append(''.join(map(str, bits)))

fano_lines = []
for i in range(7):
    for j in range(i+1, 7):
        k_vec = (fano_pts[i] + fano_pts[j]) % 2
        k_label = ''.join(map(str, k_vec.tolist()))
        if k_label in fano_labels:
            k = fano_labels.index(k_label)
            if k > j:
                line = frozenset([i, j, k])
                if line not in fano_lines:
                    fano_lines.append(line)

def fano_perm(elem_idx):
    M = elems[elem_idx]
    perm = []
    for fp in fano_pts:
        image = (M @ fp) % 2
        perm.append(fano_labels.index(''.join(map(str, image.tolist()))))
    return tuple(perm)

def line_perm(perm):
    lperm = []
    for line in fano_lines:
        image_line = frozenset(perm[p] for p in line)
        target = fano_lines.index(image_line)
        lperm.append(target)
    return tuple(lperm)

def fano_fixed_points(perm):
    return [i for i in range(7) if perm[i] == i]

def fixed_lines(lperm):
    return [i for i in range(7) if lperm[i] == i]

def fixed_flags(perm, lperm):
    flags = []
    for pi in fano_fixed_points(perm):
        for li in fixed_lines(lperm):
            if pi in fano_lines[li]:
                flags.append((pi, li))
    return flags

# PSL generators for Cayley subgraph
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

psl_gens = find_generators(mul, ords, ID)
psl_gen_set = sorted(set(psl_gens + [inv_table[g] for g in psl_gens]))

# Identify B_31 singletons (from Cayley subgraph)
b31_list = sorted(B_31)
b31_idx_map = {e: i for i, e in enumerate(b31_list)}
n_b = len(b31_list)

adj_b = [set() for _ in range(n_b)]
for i, e in enumerate(b31_list):
    for g in psl_gen_set:
        r = mul[e, g]
        if r in b31_idx_map:
            adj_b[i].add(b31_idx_map[r])
        l = mul[g, e]
        if l in b31_idx_map:
            adj_b[i].add(b31_idx_map[l])

visited = set()
components = []
for start in range(n_b):
    if start in visited:
        continue
    comp = set()
    queue = deque([start])
    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        comp.add(node)
        for nbr in adj_b[node]:
            if nbr not in visited:
                queue.append(nbr)
    components.append(comp)

singletons = sorted([b31_list[list(c)[0]] for c in components if len(c) == 1])

# Split singletons by order
flag12 = sorted([e for e in singletons if ords[e] == 4])
line9 = sorted([e for e in singletons if ords[e] == 2])

print(f"\n  B_31 singletons: {len(singletons)}")
print(f"  Order-4 flag elements: {len(flag12)} = {flag12}")
print(f"  Order-2 line involutions: {len(line9)} = {line9}")

# =========================================================================
# PART A: G_2 TEST FOR THE 12 ORDER-4 FLAG ELEMENTS
# =========================================================================
print("\n" + "=" * 72)
print("  PART A: G_2 ROOT STRUCTURE OF 12 ORDER-4 FLAG ELEMENTS")
print("=" * 72)

# A1. Charge conjugation pairing
print("\n" + "-" * 72)
print("  A1. CHARGE CONJUGATION PAIRING")
print("-" * 72)

flag12_set = set(flag12)
c_pairs_a = []
c_self_a = []
c_visited = set()

for e in flag12:
    if e in c_visited:
        continue
    c_image = mul[C_idx, mul[e, inv_table[C_idx]]]
    if c_image == e:
        c_self_a.append(e)
        c_visited.add(e)
    elif c_image in flag12_set:
        c_pairs_a.append((e, c_image))
        c_visited.add(e)
        c_visited.add(c_image)
    else:
        # C maps outside flag12
        stratum = 'B' if c_image in B_31 else ('Z' if c_image in Z_62 else 'T')
        print(f"  elem {e}: C-image = {c_image} (order {ords[c_image]}, stratum {stratum}, "
              f"singleton = {c_image in set(singletons)})")
        c_visited.add(e)

print(f"\n  C-pairs within flag12: {len(c_pairs_a)}")
for a, b in c_pairs_a:
    print(f"    {a} <-> {b}")
print(f"  Self-conjugate: {len(c_self_a)} = {c_self_a}")
print(f"  C maps outside flag12: {len(flag12) - 2*len(c_pairs_a) - len(c_self_a)}")
print(f"\n  G_2 prediction: 6 pairs + 0 self-conj = 12")
print(f"  Actual: {len(c_pairs_a)} pairs + {len(c_self_a)} self-conj = {2*len(c_pairs_a) + len(c_self_a)}")
print(f"  MATCH? {len(c_pairs_a) == 6 and len(c_self_a) == 0}")

# A2. Two root lengths (6+6 split)
print("\n" + "-" * 72)
print("  A2. TWO ROOT LENGTHS: 6+6 SPLIT?")
print("-" * 72)

# Criterion 1: fixed flag identity
print(f"\n  Criterion 1: Which flag does each element fix?")
flag_map = {}
for e in flag12:
    perm = fano_perm(e)
    lp = line_perm(perm)
    ff = fixed_flags(perm, lp)
    flag_map[e] = ff[0] if ff else None
    p, l = ff[0] if ff else (-1, -1)
    print(f"    elem {e}: fixes flag ({fano_labels[p] if p >= 0 else '?'}, L{l})")

# Group by fixed point
by_fixed_point = defaultdict(list)
by_fixed_line = defaultdict(list)
for e in flag12:
    p, l = flag_map[e]
    by_fixed_point[p].append(e)
    by_fixed_line[l].append(e)

print(f"\n  By fixed Fano point:")
for p in sorted(by_fixed_point.keys()):
    print(f"    Point {fano_labels[p]}: {len(by_fixed_point[p])} elements {by_fixed_point[p]}")

print(f"\n  By fixed Fano line:")
for l in sorted(by_fixed_line.keys()):
    pts = sorted(fano_lines[l])
    print(f"    L{l} ({{{', '.join(fano_labels[q] for q in pts)}}}): "
          f"{len(by_fixed_line[l])} elements {by_fixed_line[l]}")

# Criterion 2: S_3 orbits
print(f"\n  Criterion 2: S_3-orbit decomposition of the 12")
remaining_f12 = set(flag12)
s3_orbits_f12 = []
while remaining_f12:
    x = min(remaining_f12)
    orbit = set()
    for g in S3:
        orbit.add(mul[g, mul[x, inv_table[g]]])
    s3_orbits_f12.append(sorted(orbit & flag12_set))
    remaining_f12 -= orbit

s3_orbits_f12 = [o for o in s3_orbits_f12 if o]
s3_orbits_f12.sort(key=lambda o: (len(o), o[0]))

print(f"  S_3-orbits of flag12: {len(s3_orbits_f12)}")
for i, orb in enumerate(s3_orbits_f12):
    flags_in_orb = [flag_map[e] for e in orb]
    print(f"    Orbit {i+1} (size {len(orb)}): {orb}")
    print(f"      Flags: {[(fano_labels[p], f'L{l}') for p, l in flags_in_orb]}")

# Do the orbits split into two groups of 6?
if len(s3_orbits_f12) >= 2:
    sizes = [len(o) for o in s3_orbits_f12]
    print(f"\n  Orbit sizes: {sizes}")
    # Check if there's a natural 6+6 partition
    from itertools import combinations as combs
    for r in range(1, len(s3_orbits_f12)):
        for subset in combs(range(len(s3_orbits_f12)), r):
            total = sum(sizes[i] for i in subset)
            if total == 6:
                complement = [i for i in range(len(s3_orbits_f12)) if i not in subset]
                comp_total = sum(sizes[i] for i in complement)
                if comp_total == 6:
                    print(f"  6+6 partition found:")
                    print(f"    Group A: orbits {[i+1 for i in subset]} = "
                          f"{sum(sizes[i] for i in subset)} elements")
                    print(f"    Group B: orbits {[i+1 for i in complement]} = "
                          f"{sum(sizes[i] for i in complement)} elements")

# Criterion 3: squaring target
print(f"\n  Criterion 3: Squaring targets")
sq_groups = defaultdict(list)
for e in flag12:
    sq = mul[e, e]
    sq_groups[int(sq)].append(e)

for sq_target, elems_list in sorted(sq_groups.items()):
    stratum = 'B' if sq_target in B_31 else ('Z' if sq_target in Z_62 else ('T' if sq_target in T_75 else 'S3?'))
    print(f"    Square = {sq_target} (order {ords[sq_target]}, stratum {stratum}): "
          f"{len(elems_list)} elements {elems_list}")

# A3. Triple bond structure
print("\n" + "-" * 72)
print("  A3. TRIPLE BOND STRUCTURE (3-FOLD S_3 ACTION)")
print("-" * 72)

# For each order-3 element of S_3, compute its action on the 12 flag elements
for z3_elem in s3_order3:
    print(f"\n  S_3 order-3 element {z3_elem}:")
    orbit_map = {}
    for e in flag12:
        image = mul[z3_elem, mul[e, inv_table[z3_elem]]]
        orbit_map[e] = image
        in_f12 = image in flag12_set
        print(f"    {e} -> {image} ({'IN' if in_f12 else 'OUT'} flag12, order {ords[image]})")

# Product structure between flag12 elements
print(f"\n  Product matrix (flag12 * flag12, landing stratum):")
prod_in_flag = 0
prod_in_b31 = 0
prod_in_z62 = 0
prod_in_t75 = 0
for a in flag12:
    for b in flag12:
        p = mul[a, b]
        if p in flag12_set:
            prod_in_flag += 1
        elif p in B_31:
            prod_in_b31 += 1
        elif p in Z_62:
            prod_in_z62 += 1
        elif p in T_75:
            prod_in_t75 += 1

total_p = len(flag12)**2
print(f"  flag12: {prod_in_flag} ({prod_in_flag/total_p*100:.1f}%)")
print(f"  other B_31: {prod_in_b31} ({prod_in_b31/total_p*100:.1f}%)")
print(f"  Z_62: {prod_in_z62} ({prod_in_z62/total_p*100:.1f}%)")
print(f"  T_75: {prod_in_t75} ({prod_in_t75/total_p*100:.1f}%)")

# =========================================================================
# PART B: U(3) TEST FOR THE 9 LINE INVOLUTIONS
# =========================================================================
print("\n" + "=" * 72)
print("  PART B: U(3) STRUCTURE OF 9 ORDER-2 LINE INVOLUTIONS")
print("=" * 72)

# B1. Fixed lines
print("\n" + "-" * 72)
print("  B1. FIXED FANO LINES OF EACH INVOLUTION")
print("-" * 72)

inv_fixed_lines_map = {}
for e in line9:
    perm = fano_perm(e)
    lp = line_perm(perm)
    fix_pts = fano_fixed_points(perm)
    fix_lns = fixed_lines(lp)
    inv_fixed_lines_map[e] = (fix_pts, fix_lns)
    print(f"  elem {e}: fixes points {[fano_labels[p] for p in fix_pts]}, "
          f"lines {['L'+str(l) for l in fix_lns]}")

# Which line contains the 3 fixed points?
print(f"\n  Fixed-point line (the line spanned by the 3 fixed points):")
inv_to_fixline = {}
for e in line9:
    fix_pts = inv_fixed_lines_map[e][0]
    # Find the Fano line containing all 3 fixed points
    for li, line in enumerate(fano_lines):
        if set(fix_pts) == set(line):
            inv_to_fixline[e] = li
            pts_str = ', '.join(fano_labels[p] for p in sorted(line))
            print(f"  elem {e}: fixed-point line = L{li} ({{{pts_str}}})")
            break
    else:
        # The 3 fixed points may not all be collinear!
        inv_to_fixline[e] = None
        print(f"  elem {e}: fixed points {[fano_labels[p] for p in fix_pts]} NOT collinear!")

# Distribution over lines
line_count = Counter(inv_to_fixline.values())
print(f"\n  Line distribution (which lines have how many involutions):")
for l in range(7):
    pts = sorted(fano_lines[l])
    n = line_count.get(l, 0)
    invs = [e for e in line9 if inv_to_fixline[e] == l]
    print(f"    L{l} ({{{', '.join(fano_labels[p] for p in pts)}}}): {n} involutions {invs}")

non_collinear = [e for e in line9 if inv_to_fixline[e] is None]
if non_collinear:
    print(f"  Non-collinear fixed points: {len(non_collinear)} involutions")

# B2. S_3 action on the 9 line involutions
print("\n" + "-" * 72)
print("  B2. S_3 ACTION ON THE 9 LINE INVOLUTIONS")
print("-" * 72)

remaining_l9 = set(line9)
s3_orbits_l9 = []
while remaining_l9:
    x = min(remaining_l9)
    orbit = set()
    for g in S3:
        img = mul[g, mul[x, inv_table[g]]]
        orbit.add(img)
    # Intersect with line9
    orbit_in_l9 = orbit & set(line9)
    s3_orbits_l9.append(sorted(orbit_in_l9))
    remaining_l9 -= orbit_in_l9

s3_orbits_l9 = [o for o in s3_orbits_l9 if o]
s3_orbits_l9.sort(key=lambda o: (len(o), o[0]))

print(f"\n  S_3-orbits of line9: {len(s3_orbits_l9)}")
for i, orb in enumerate(s3_orbits_l9):
    fix_lns = [inv_to_fixline[e] for e in orb]
    print(f"    Orbit {i+1} (size {len(orb)}): {orb}, fixed lines = {fix_lns}")

# S_3 singlet test: is any line involution fixed by ALL of S_3?
print(f"\n  S_3 singlet test (photon candidate):")
for e in line9:
    fixed_by_all = all(mul[g, mul[e, inv_table[g]]] == e for g in S3)
    if fixed_by_all:
        print(f"    elem {e}: FIXED BY ALL S_3 (singlet!) "
              f"-- fixes {[fano_labels[p] for p in inv_fixed_lines_map[e][0]]}")

# Check each S_3 element individually
for e in line9:
    fixers = [g for g in S3 if mul[g, mul[e, inv_table[g]]] == e]
    if len(fixers) > 1:  # more than just identity
        print(f"    elem {e}: fixed by {len(fixers)}/{len(S3)} S_3 elements")

# B3. Commutator structure
print("\n" + "-" * 72)
print("  B3. COMMUTATOR STRUCTURE OF 9 LINE INVOLUTIONS")
print("-" * 72)

print(f"\n  [inv_i, inv_j] = inv_i * inv_j * inv_i^-1 * inv_j^-1")
print(f"  (For involutions, inv^-1 = inv, so [a,b] = a*b*a*b)")
print(f"\n  Commutator matrix (element indices):")
print(f"  {'':>6}", end='')
for b in line9:
    print(f" {b:>5}", end='')
print()

comm_matrix = np.zeros((9, 9), dtype=int)
comm_orders = np.zeros((9, 9), dtype=int)
for i, a in enumerate(line9):
    print(f"  {a:>6}", end='')
    for j, b in enumerate(line9):
        comm = mul[a, mul[b, mul[a, b]]]  # a*b*a*b since a^-1=a, b^-1=b
        comm_matrix[i, j] = comm
        comm_orders[i, j] = ords[comm]
        print(f" {comm:>5}", end='')
    print()

print(f"\n  Commutator order matrix:")
print(f"  {'':>6}", end='')
for b in line9:
    print(f" {b:>5}", end='')
print()
for i, a in enumerate(line9):
    print(f"  {a:>6}", end='')
    for j, b in enumerate(line9):
        print(f" {comm_orders[i,j]:>5}", end='')
    print()

# Where do commutators land?
comm_strata = Counter()
for i in range(9):
    for j in range(9):
        if i == j:
            continue
        c = comm_matrix[i, j]
        if c == ID:
            comm_strata['ID'] += 1
        elif c in B_31:
            comm_strata['B'] += 1
        elif c in Z_62:
            comm_strata['Z'] += 1
        elif c in T_75:
            comm_strata['T'] += 1

print(f"\n  Commutator landing (off-diagonal):")
for s in ['ID', 'B', 'Z', 'T']:
    if s in comm_strata:
        print(f"    {s}: {comm_strata[s]}")

# How many pairs commute?
n_commuting = sum(1 for i in range(9) for j in range(i+1, 9) if comm_matrix[i,j] == ID)
print(f"\n  Commuting pairs: {n_commuting}/{9*8//2}")
print(f"  Non-commuting pairs: {9*8//2 - n_commuting}/{9*8//2}")

# SU(3) has rank 2: exactly 2 mutually commuting generators (Cartan subalgebra)
# Find maximal set of mutually commuting line involutions
print(f"\n  Maximal commuting subsets:")
for size in range(9, 0, -1):
    found = False
    for subset in combinations(range(9), size):
        if all(comm_matrix[i, j] == ID for i in subset for j in subset):
            found = True
            print(f"    Size {size}: {[line9[i] for i in subset]}")
            if size <= 4:
                break
    if found and size <= 4:
        break

# =========================================================================
# PART C: SQUARING MAP
# =========================================================================
print("\n" + "=" * 72)
print("  PART C: SQUARING MAP (ORDER-4 -> INVOLUTIONS)")
print("=" * 72)

print(f"\n  {'Elem':>5} {'Square':>7} {'Sq_ord':>7} {'Sq_stratum':>11} "
      f"{'Sq_in_B31':>10} {'Sq_singleton':>13} {'Flag'}")
print(f"  {'-'*5} {'-'*7} {'-'*7} {'-'*11} {'-'*10} {'-'*13} {'-'*15}")

sq_in_b31 = []
sq_in_t75 = []
sq_in_s3 = []
sq_other = []

for e in flag12:
    sq = mul[e, e]
    sq_ord = int(ords[sq])
    if sq in B_31:
        stratum = 'B_31'
        sq_in_b31.append(e)
    elif sq in Z_62:
        stratum = 'Z_62'
        sq_other.append(e)
    elif sq in T_75:
        stratum = 'T_75'
        sq_in_t75.append(e)
    else:
        stratum = '???'
        sq_other.append(e)

    is_s3 = sq in set(S3)
    if is_s3:
        sq_in_s3.append(e)

    is_sing = sq in set(singletons)
    p, l = flag_map.get(e, (-1, -1)) if hasattr(flag_map.get(e, None), '__iter__') else (-1, -1)
    p, l = flag_map[e]
    flag_str = f"({fano_labels[p]},L{l})"

    print(f"  {e:5d} {sq:7d} {sq_ord:7d} {stratum:>11} "
          f"{'YES' if sq in B_31 else 'NO':>10} "
          f"{'YES' if is_sing else 'NO':>13} {flag_str}")

print(f"\n  Squares landing in B_31: {len(sq_in_b31)} elements {sq_in_b31}")
print(f"  Squares landing in T_75: {len(sq_in_t75)} elements {sq_in_t75}")
print(f"  Squares that are S_3 elements: {len(sq_in_s3)} elements {sq_in_s3}")

# Identify the squares outside B_31
print(f"\n  Non-B_31 squares (identifying strata):")
all_squares = set()
for e in flag12:
    sq = mul[e, e]
    all_squares.add(int(sq))

for sq in sorted(all_squares):
    stratum = 'B_31' if sq in B_31 else ('Z_62' if sq in Z_62 else ('T_75' if sq in T_75 else '???'))
    is_s3 = sq in set(S3)
    is_fix_c = sq in fix_C
    sources = [e for e in flag12 if mul[e, e] == sq]
    print(f"  Square {sq}: order {ords[sq]}, stratum {stratum}, "
          f"S_3={'YES' if is_s3 else 'NO'}, Fix(C)={'YES' if is_fix_c else 'NO'}, "
          f"sources {sources}")

# Elements 0, 78, 100 specifically
print(f"\n  Specific non-B_31 square targets:")
for target in [0, 78, 100]:
    if target < 168:
        stratum = 'B_31' if target in B_31 else ('Z_62' if target in Z_62 else ('T_75' if target in T_75 else '???'))
        is_s3 = target in set(S3)
        print(f"    Element {target}: order {ords[target]}, stratum {stratum}, "
              f"S_3={'YES' if is_s3 else 'NO'}, in B_31={target in B_31}")

# =========================================================================
# PART D: COMBINED STRUCTURE
# =========================================================================
print("\n" + "=" * 72)
print("  PART D: COMBINED 12 + 9 = 21 STRUCTURE")
print("=" * 72)

# The 21 singletons as a whole: how do they interact?
sing_set = set(singletons)

# Products within the 21
print(f"\n  Products within 21 singletons:")
sing_prod = Counter()
for a in singletons:
    for b in singletons:
        p = mul[a, b]
        if p in sing_set:
            sing_prod['singleton'] += 1
        elif p in B_31:
            sing_prod['other_B'] += 1
        elif p in Z_62:
            sing_prod['Z'] += 1
        elif p in T_75:
            sing_prod['T'] += 1

total_sp = len(singletons)**2
for k, v in sorted(sing_prod.items()):
    print(f"    {k}: {v} ({v/total_sp*100:.1f}%)")

# Cross products: flag12 * line9
print(f"\n  Cross products flag12 * line9:")
cross = Counter()
for a in flag12:
    for b in line9:
        p = mul[a, b]
        if p in flag12_set:
            cross['flag12'] += 1
        elif p in set(line9):
            cross['line9'] += 1
        elif p in B_31:
            cross['other_B'] += 1
        elif p in Z_62:
            cross['Z'] += 1
        elif p in T_75:
            cross['T'] += 1

total_cr = len(flag12) * len(line9)
for k, v in sorted(cross.items()):
    print(f"    {k}: {v} ({v/total_cr*100:.1f}%)")

# =========================================================================
# SUMMARY
# =========================================================================
print("\n" + "=" * 72)
print("  SUMMARY")
print("=" * 72)

print(f"""
  PART A: G_2 ROOT STRUCTURE (12 order-4 flag elements)
    C-pairs within flag12: {len(c_pairs_a)}
    Self-conjugate: {len(c_self_a)}
    G_2 requires: 6 pairs + 0 self-conj = 12
    MATCH: {len(c_pairs_a) == 6 and len(c_self_a) == 0}
    S_3-orbit decomposition: {[len(o) for o in s3_orbits_f12]}

  PART B: U(3) STRUCTURE (9 order-2 line involutions)
    S_3-orbits: {[len(o) for o in s3_orbits_l9]}
    S_3-singlet (photon): {'found' if any(all(mul[g, mul[e, inv_table[g]]] == e for g in S3) for e in line9) else 'NONE'}
    Non-commuting pairs: {9*8//2 - n_commuting}/{9*8//2}

  PART C: SQUARING MAP
    Squares in B_31: {len(sq_in_b31)}/12
    Squares in S_3: {len(sq_in_s3)}/12
    Squares in T_75: {len(sq_in_t75)}/12
""")
