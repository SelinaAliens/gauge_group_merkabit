"""
lie_algebra_matter.py
Verify Lie algebra structure of the 9 line involutions and identify
the full commutator algebra. Test Cartan structure, S_3 stratum
distribution, and geometric stability criterion.
"""
import sys
import numpy as np
from collections import Counter, defaultdict, deque
from itertools import product as iproduct, combinations

sys.path.insert(0, '.')
from psl27_core import build_group, classify_strata, conjugacy_classes, mat_key

print("=" * 72)
print("  LIE ALGEBRA OF MATTER: COMMUTATOR ALGEBRA + STABILITY")
print("=" * 72)

elems, e2i, mul, inv_table, ords, ID = build_group()
B_31, Z_62, T_75, z3, z3sq, W26, S3 = classify_strata(
    elems, e2i, mul, inv_table, ords, ID)

antidiag = np.array([[0,0,1],[0,1,0],[1,0,0]], dtype=int)
C_idx = e2i[mat_key(antidiag)]
fix_C = {i for i in range(168) if mul[C_idx, mul[i, inv_table[C_idx]]] == i}

F36 = sorted(Z_62 - W26)
F36_set = set(F36)
W26_set = set(W26)
S3_set = set(S3)

# Fano
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

def fano_cycle_type(perm):
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

# Singletons from Cayley subgraph
psl_gens_pair = None
for a in range(168):
    if ords[a] != 2: continue
    for b in range(168):
        if ords[b] != 3: continue
        generated = {ID}
        frontier = {ID, a, b}
        while frontier - generated:
            generated.update(frontier)
            new = set()
            for x in frontier:
                for g in [a, b]:
                    new.add(mul[x, g])
                    new.add(mul[g, x])
            frontier = new - generated
        if len(generated) == 168:
            psl_gens_pair = [a, b]
            break
    if psl_gens_pair:
        break

psl_gen_set = sorted(set(psl_gens_pair + [inv_table[g] for g in psl_gens_pair]))

b31_list = sorted(B_31)
b31_idx_map = {e: i for i, e in enumerate(b31_list)}
adj_b = [set() for _ in range(len(b31_list))]
for i, e in enumerate(b31_list):
    for g in psl_gen_set:
        for nxt in [mul[e, g], mul[g, e]]:
            if nxt in b31_idx_map:
                adj_b[i].add(b31_idx_map[nxt])

visited_b = set()
components = []
for start in range(len(b31_list)):
    if start in visited_b: continue
    comp = set()
    queue = deque([start])
    while queue:
        node = queue.popleft()
        if node in visited_b: continue
        visited_b.add(node)
        comp.add(node)
        for nbr in adj_b[node]:
            if nbr not in visited_b:
                queue.append(nbr)
    components.append(comp)

singletons = sorted([b31_list[list(c)[0]] for c in components if len(c) == 1])
flag12 = sorted([e for e in singletons if ords[e] == 4])
line9 = sorted([e for e in singletons if ords[e] == 2])

print(f"\n  flag12: {flag12}")
print(f"  line9: {line9}")

# =========================================================================
# PART A: COMMUTATOR ALGEBRA
# =========================================================================
print("\n" + "=" * 72)
print("  PART A: COMMUTATOR ALGEBRA OF 9 LINE INVOLUTIONS")
print("=" * 72)

# For involutions, [a,b] = a*b*a*b (since a^-1=a, b^-1=b)
print("\n" + "-" * 72)
print("  A1. ALL 72 ORDERED COMMUTATORS")
print("-" * 72)

comm_results = {}
for a in line9:
    for b in line9:
        if a == b:
            comm_results[(a,b)] = ID
        else:
            c = mul[a, mul[b, mul[a, b]]]
            comm_results[(a,b)] = c

# Classify each commutator
print(f"\n  {'[a,b]':>12} {'Result':>7} {'Order':>6} {'Stratum':>8} {'Sector':>8}")
print(f"  {'-'*12} {'-'*7} {'-'*6} {'-'*8} {'-'*8}")

comm_sector_count = Counter()
comm_distinct = set()
for a in line9:
    for b in line9:
        if a == b: continue
        c = comm_results[(a,b)]
        comm_distinct.add(c)
        stratum = 'B' if c in B_31 else ('Z' if c in Z_62 else 'T')
        if c == ID:
            sector = 'ID'
        elif c in W26_set:
            sector = 'W26'
        elif c in F36_set:
            sector = 'F36'
        elif c in B_31:
            sector = 'B_other'
        elif c in T_75:
            sector = 'T'
        else:
            sector = '?'
        comm_sector_count[sector] += 1

print(f"\n  Commutator sector distribution (72 ordered, off-diagonal):")
for s in ['ID', 'W26', 'F36', 'B_other', 'T']:
    if s in comm_sector_count:
        print(f"    {s}: {comm_sector_count[s]}")

print(f"\n  Distinct commutator elements: {len(comm_distinct)}")
print(f"  Distinct non-identity: {len(comm_distinct - {ID})}")

# Classify distinct commutators
print(f"\n  Distinct commutators:")
for c in sorted(comm_distinct):
    if c == ID: continue
    stratum = 'B' if c in B_31 else ('Z' if c in Z_62 else 'T')
    sector = 'W26' if c in W26_set else ('F36' if c in F36_set else ('S3' if c in S3_set else stratum))
    print(f"    elem {c}: order {ords[c]}, stratum {stratum}, sector {sector}")

# A2. Algebra closure
print("\n" + "-" * 72)
print("  A2. ALGEBRA GENERATED BY REPEATED COMMUTATORS")
print("-" * 72)

# Start with line9, take commutators, add to set, repeat
algebra = set(line9)
algebra.add(ID)
step = 0
while True:
    new_elems = set()
    for a in algebra:
        for b in algebra:
            # Product a*b
            new_elems.add(mul[a, b])
            # Commutator a*b*a^{-1}*b^{-1}
            c = mul[a, mul[b, mul[inv_table[a], inv_table[b]]]]
            new_elems.add(c)
    if new_elems <= algebra:
        break
    algebra.update(new_elems)
    step += 1
    in_b = sum(1 for e in algebra if e in B_31)
    in_z = sum(1 for e in algebra if e in Z_62)
    in_t = sum(1 for e in algebra if e in T_75)
    print(f"  Step {step}: |algebra| = {len(algebra)} (B={in_b}, Z={in_z}, T={in_t})")

print(f"\n  Final algebra size: {len(algebra)}")
print(f"  Is full PSL(2,7)? {len(algebra) == 168}")

# What subgroup do the 9 line involutions generate (as a group under multiplication)?
gen9 = set(line9) | {ID}
frontier9 = set(gen9)
while True:
    new9 = set()
    for a in frontier9:
        for b in gen9:
            new9.add(mul[a, b])
            new9.add(mul[b, a])
    if new9 <= gen9:
        break
    frontier9 = new9 - gen9
    gen9.update(new9)

print(f"\n  <line9> (group generated by multiplication): {len(gen9)} elements")
print(f"  Is full PSL(2,7)? {len(gen9) == 168}")

# Stratum distribution of <line9>
g9_b = sum(1 for e in gen9 if e in B_31)
g9_z = sum(1 for e in gen9 if e in Z_62)
g9_t = sum(1 for e in gen9 if e in T_75)
print(f"  Stratum: B={g9_b}, Z={g9_z}, T={g9_t}")

# =========================================================================
# PART B: CARTAN STRUCTURE
# =========================================================================
print("\n" + "=" * 72)
print("  PART B: CARTAN SUBALGEBRA {77, 124, 125}")
print("=" * 72)

cartan = [77, 124, 125]

# B1. Verify mutual commutation
print("\n" + "-" * 72)
print("  B1. MUTUAL COMMUTATION")
print("-" * 72)

for a, b in combinations(cartan, 2):
    c = mul[a, mul[b, mul[a, b]]]
    print(f"  [{a}, {b}] = {c} (= ID? {c == ID})")

# B2. Fixed structure
print("\n" + "-" * 72)
print("  B2. FANO FIXED STRUCTURE")
print("-" * 72)

# L1 = {001, 100, 101} = fano_lines[1]
L1_pts = sorted(fano_lines[1])
L1_labels = [fano_labels[p] for p in L1_pts]
print(f"\n  L1 = {{{', '.join(L1_labels)}}}")

for e in cartan:
    perm = fano_perm(e)
    lp = line_perm(perm)
    fix_pts = fano_fixed_points(perm)
    fix_lns = fixed_lines(lp)
    print(f"  elem {e}: fixes pts {[fano_labels[p] for p in fix_pts]}, "
          f"lines {['L'+str(l) for l in fix_lns]}")

# Joint fixed points
joint_fixed = None
for e in cartan:
    perm = fano_perm(e)
    fps = set(fano_fixed_points(perm))
    if joint_fixed is None:
        joint_fixed = fps
    else:
        joint_fixed &= fps

print(f"\n  Joint fixed points of {{77, 124, 125}}: {[fano_labels[p] for p in sorted(joint_fixed)]}")
print(f"  These are exactly L1? {joint_fixed == set(L1_pts)}")

# B3. Eisenstein direction
print(f"\n  010 (Eisenstein/element-18 direction):")
pt_010 = fano_labels.index('010')
print(f"  010 on L1? {pt_010 in fano_lines[1]}")

# Complement of L1
complement = [i for i in range(7) if i not in fano_lines[1]]
print(f"\n  Complement of L1: {[fano_labels[p] for p in complement]}")
print(f"  (4 points: {len(complement)})")

# Do these 4 points form any Fano lines?
comp_lines = []
for line in fano_lines:
    if line <= set(complement):
        comp_lines.append(line)
print(f"  Fano lines within complement: {len(comp_lines)}")
for cl in comp_lines:
    print(f"    {{{', '.join(fano_labels[p] for p in sorted(cl))}}}")

# B4. Group generated by {77, 124, 125}
print("\n" + "-" * 72)
print("  B4. GROUP GENERATED BY CARTAN ELEMENTS")
print("-" * 72)

cartan_group = {ID}
frontier_c = set(cartan) | {ID}
while frontier_c - cartan_group:
    cartan_group.update(frontier_c)
    new_c = set()
    for a in frontier_c:
        for b in cartan:
            new_c.add(mul[a, b])
            new_c.add(mul[b, a])
    frontier_c = new_c - cartan_group

print(f"  |<77, 124, 125>| = {len(cartan_group)}")
cg_orders = Counter(int(ords[e]) for e in cartan_group)
print(f"  Order distribution: {dict(sorted(cg_orders.items()))}")
print(f"  Elements: {sorted(cartan_group)}")

# Is it Z_2^3?
is_all_involutions = all(ords[e] in [1, 2] for e in cartan_group)
print(f"  All elements order 1 or 2? {is_all_involutions}")
if len(cartan_group) == 8 and is_all_involutions:
    print(f"  Isomorphic to Z_2^3 (elementary abelian of rank 3)? YES")
elif len(cartan_group) == 4 and is_all_involutions:
    print(f"  Isomorphic to Klein four-group V_4? YES")
else:
    print(f"  Structure: order {len(cartan_group)}, "
          f"{'abelian' if all(mul[a,b]==mul[b,a] for a in cartan_group for b in cartan_group) else 'non-abelian'}")

# Which stratum?
cg_strata = Counter()
for e in cartan_group:
    if e in B_31: cg_strata['B'] += 1
    elif e in Z_62: cg_strata['Z'] += 1
    elif e in T_75: cg_strata['T'] += 1
print(f"  Stratum distribution: {dict(cg_strata)}")

# =========================================================================
# PART C: S_3 STRATUM DISTRIBUTION
# =========================================================================
print("\n" + "=" * 72)
print("  PART C: COMPLETE S_3 ELEMENT TABLE")
print("=" * 72)

print(f"\n  {'Elem':>5} {'Order':>6} {'Stratum':>8} {'Sector':>8} {'PtCycle':>15} "
      f"{'FixPts':>20} {'FixLns':>15}")
print(f"  {'-'*5} {'-'*6} {'-'*8} {'-'*8} {'-'*15} {'-'*20} {'-'*15}")

for g in sorted(S3):
    stratum = 'B_31' if g in B_31 else ('Z_62' if g in Z_62 else 'T_75')
    sector = stratum
    if g in W26_set: sector = 'W26'
    elif g in F36_set: sector = 'F36'
    perm = fano_perm(g)
    ct = fano_cycle_type(perm)
    lp = line_perm(perm)
    fps = [fano_labels[p] for p in fano_fixed_points(perm)]
    fls = ['L'+str(l) for l in fixed_lines(lp)]
    print(f"  {g:5d} {ords[g]:6d} {stratum:>8} {sector:>8} {str(ct):>15} "
          f"{str(fps):>20} {str(fls):>15}")

# Verify S_3 involutions
s3_inv = [g for g in S3 if ords[g] == 2]
s3_ord3 = [g for g in S3 if ords[g] == 3]
print(f"\n  S_3 involutions: {s3_inv}")
for g in s3_inv:
    stratum = 'B_31' if g in B_31 else ('Z_62' if g in Z_62 else 'T_75')
    print(f"    elem {g}: stratum {stratum}, in T_75? {g in T_75}")

print(f"\n  S_3 order-3 elements: {s3_ord3}")
for g in s3_ord3:
    stratum = 'B_31' if g in B_31 else ('Z_62' if g in Z_62 else 'T_75')
    in_w26 = g in W26_set
    in_f36 = g in F36_set
    print(f"    elem {g}: stratum {stratum}, in W26? {in_w26}, in F36? {in_f36}")

# Verify S_3 preserves all strata
print(f"\n  S_3 stratum preservation:")
for g in S3:
    pres_b = all(mul[g, mul[b, inv_table[g]]] in B_31 for b in B_31)
    pres_z = all(mul[g, mul[z, inv_table[g]]] in Z_62 for z in Z_62)
    pres_t = all(mul[g, mul[t, inv_table[g]]] in T_75 for t in T_75)
    print(f"    elem {g} (order {ords[g]}): B={pres_b}, Z={pres_z}, T={pres_t}")

# =========================================================================
# PART D: STABILITY CRITERION
# =========================================================================
print("\n" + "=" * 72)
print("  PART D: GEOMETRIC STABILITY CRITERION")
print("=" * 72)

# Flag data for the 12 order-4 flag elements
def get_fixed_flag(e):
    perm = fano_perm(e)
    lp = line_perm(perm)
    for pi in fano_fixed_points(perm):
        for li in fixed_lines(lp):
            if pi in fano_lines[li]:
                return (pi, li)
    return None

# D1. Stable vs unstable
print("\n" + "-" * 72)
print("  D1. STABILITY TABLE")
print("-" * 72)

stable6 = sorted([e for e in flag12 if mul[e,e] in B_31])
unstable6 = sorted([e for e in flag12 if mul[e,e] in T_75])

print(f"\n  L1 = line 1 = {{{', '.join(fano_labels[p] for p in sorted(fano_lines[1]))}}} = {{001, 100, 101}}")
print(f"\n  {'Elem':>5} {'Stable':>7} {'Flag(pt,ln)':>15} {'Pt on L1':>9} {'Ln=L1':>6} "
      f"{'Square':>7} {'Sq_stratum':>11} {'Sq_is_S3':>9}")
print(f"  {'-'*5} {'-'*7} {'-'*15} {'-'*9} {'-'*6} {'-'*7} {'-'*11} {'-'*9}")

for e in flag12:
    sq = mul[e, e]
    is_stable = sq in B_31
    flag = get_fixed_flag(e)
    p, l = flag
    pt_on_L1 = p in fano_lines[1]
    ln_is_L1 = l == 1
    sq_stratum = 'B_31' if sq in B_31 else ('T_75' if sq in T_75 else 'Z_62')
    sq_is_s3 = sq in S3_set
    flag_str = f"({fano_labels[p]},L{l})"
    print(f"  {e:5d} {'YES' if is_stable else 'NO':>7} {flag_str:>15} "
          f"{'YES' if pt_on_L1 else 'NO':>9} {'YES' if ln_is_L1 else 'NO':>6} "
          f"{sq:7d} {sq_stratum:>11} {'YES' if sq_is_s3 else 'NO':>9}")

# D2. Test the L1 criterion
print("\n" + "-" * 72)
print("  D2. L1 STABILITY CRITERION TEST")
print("-" * 72)

stable_on_L1 = sum(1 for e in stable6 if get_fixed_flag(e)[0] in fano_lines[1])
stable_off_L1 = len(stable6) - stable_on_L1
unstable_on_L1 = sum(1 for e in unstable6 if get_fixed_flag(e)[0] in fano_lines[1])
unstable_off_L1 = len(unstable6) - unstable_on_L1

print(f"\n  Stable elements with fixed point ON L1:  {stable_on_L1}/{len(stable6)}")
print(f"  Stable elements with fixed point OFF L1: {stable_off_L1}/{len(stable6)}")
print(f"  Unstable elements with fixed point ON L1:  {unstable_on_L1}/{len(unstable6)}")
print(f"  Unstable elements with fixed point OFF L1: {unstable_off_L1}/{len(unstable6)}")

criterion_holds = (stable_on_L1 == 0 and unstable_on_L1 == len(unstable6)) or \
                  (stable_on_L1 == len(stable6) and unstable_off_L1 == len(unstable6))
print(f"\n  L1 criterion (stable ON, unstable OFF, or vice versa): {criterion_holds}")

# Alternative: test fixed LINE = L1
stable_ln_L1 = sum(1 for e in stable6 if get_fixed_flag(e)[1] == 1)
unstable_ln_L1 = sum(1 for e in unstable6 if get_fixed_flag(e)[1] == 1)
print(f"\n  Fixed line = L1:")
print(f"    Stable: {stable_ln_L1}/{len(stable6)}")
print(f"    Unstable: {unstable_ln_L1}/{len(unstable6)}")

# D3. Full flag-line analysis
print("\n" + "-" * 72)
print("  D3. FLAG LINES OF STABLE vs UNSTABLE")
print("-" * 72)

print(f"\n  Stable flag elements (square in B_31):")
stable_lines = Counter()
for e in stable6:
    p, l = get_fixed_flag(e)
    stable_lines[l] += 1
    print(f"    elem {e}: flag ({fano_labels[p]}, L{l})")
print(f"  Fixed lines: {dict(stable_lines)}")

print(f"\n  Unstable flag elements (square in T_75 via S_3):")
unstable_lines = Counter()
for e in unstable6:
    p, l = get_fixed_flag(e)
    unstable_lines[l] += 1
    print(f"    elem {e}: flag ({fano_labels[p]}, L{l})")
print(f"  Fixed lines: {dict(unstable_lines)}")

# Is there ANY line that separates stable from unstable?
print(f"\n  Line separation test (is any line purely stable or purely unstable?):")
for li in range(7):
    s_count = sum(1 for e in stable6 if get_fixed_flag(e)[1] == li)
    u_count = sum(1 for e in unstable6 if get_fixed_flag(e)[1] == li)
    marker = ""
    if s_count > 0 and u_count == 0: marker = " <-- PURELY STABLE"
    elif u_count > 0 and s_count == 0: marker = " <-- PURELY UNSTABLE"
    pts = sorted(fano_lines[li])
    print(f"    L{li} ({{{', '.join(fano_labels[p] for p in pts)}}}): "
          f"stable={s_count}, unstable={u_count}{marker}")

# D4. Square targets detail
print("\n" + "-" * 72)
print("  D4. SQUARE TARGETS IN DETAIL")
print("-" * 72)

print(f"\n  Stable squares (in B_31):")
for e in stable6:
    sq = mul[e, e]
    is_line9 = sq in set(line9)
    is_singleton = sq in set(singletons)
    print(f"    {e}^2 = {sq} (order {ords[sq]}, "
          f"line involution? {is_line9}, singleton? {is_singleton})")

print(f"\n  Unstable squares (in T_75, S_3 involutions):")
for e in unstable6:
    sq = mul[e, e]
    print(f"    {e}^2 = {sq} (order {ords[sq]}, S_3? {sq in S3_set})")

# Which S_3 involutions are hit?
unstable_sq_targets = set(mul[e,e] for e in unstable6)
print(f"\n  S_3 involutions hit by unstable squares: {sorted(unstable_sq_targets)}")
print(f"  All 3 S_3 involutions: {sorted(s3_inv)}")
print(f"  All 3 hit? {unstable_sq_targets == set(s3_inv)}")

# =========================================================================
# SUMMARY
# =========================================================================
print("\n" + "=" * 72)
print("  SUMMARY")
print("=" * 72)

print(f"""
  PART A: COMMUTATOR ALGEBRA
    Distinct non-ID commutators: {len(comm_distinct - {ID})}
    Landing: {dict(comm_sector_count)}
    <line9> generates: {len(gen9)} elements (full PSL? {len(gen9)==168})

  PART B: CARTAN SUBALGEBRA {{77, 124, 125}}
    Mutual commutation: verified
    Joint fixed points = L1 = {{001, 100, 101}}: {joint_fixed == set(L1_pts)}
    010 on L1? {pt_010 in fano_lines[1]}
    |<Cartan>| = {len(cartan_group)}, orders = {dict(sorted(cg_orders.items()))}

  PART C: S_3 DISTRIBUTION
    S_3 involutions in T_75: {all(g in T_75 for g in s3_inv)}
    S_3 order-3 in Z_62: {all(g in Z_62 for g in s3_ord3)}

  PART D: STABILITY CRITERION
    Stable (sq in B_31): {len(stable6)} flag elements
    Unstable (sq in T_75): {len(unstable6)} flag elements
    L1 separates stable/unstable: {criterion_holds}
    Stable lines: {dict(stable_lines)}
    Unstable lines: {dict(unstable_lines)}
""")
