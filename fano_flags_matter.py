"""
fano_flags_matter.py
Test whether the 21 singleton matter elements correspond to the 21 Fano flags.

A Fano flag = incident (point, line) pair. 7 lines x 3 points/line = 21 flags.
B_31 has 25 components under PSL-generator Cayley subgraph:
  21 singletons + 2 triples + 2 doubles = 25 components, 31 elements.

Questions:
1. Do the 21 singletons biject to the 21 Fano flags?
2. What Fano geometry do the 10 non-singleton elements encode?
3. What do the triples and doubles correspond to?
"""
import sys
import numpy as np
from collections import Counter, defaultdict, deque
from itertools import product as iproduct

sys.path.insert(0, '.')
from psl27_core import build_group, classify_strata, mat_key

print("=" * 72)
print("  FANO FLAGS AND MATTER: B_31 SINGLETON CORRESPONDENCE")
print("=" * 72)

elems, e2i, mul, inv_table, ords, ID = build_group()
B_31, Z_62, T_75, z3, z3sq, W26, S3 = classify_strata(
    elems, e2i, mul, inv_table, ords, ID)

# =========================================================================
# FANO PLANE INFRASTRUCTURE
# =========================================================================

# 7 Fano points = nonzero vectors in F_2^3
fano_pts = []
fano_labels = []
for bits in iproduct([0,1], repeat=3):
    if any(b for b in bits):
        fano_pts.append(np.array(bits, dtype=int))
        fano_labels.append(''.join(map(str, bits)))

# 7 Fano lines: each line is a set of 3 collinear points (vectors that sum to 0 mod 2)
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

assert len(fano_lines) == 7, f"Expected 7 Fano lines, got {len(fano_lines)}"

print(f"\n  Fano plane: {len(fano_pts)} points, {len(fano_lines)} lines")
print(f"  Points: {fano_labels}")
print(f"  Lines:")
for li, line in enumerate(fano_lines):
    pts = sorted(line)
    print(f"    L{li}: {{{', '.join(fano_labels[p] for p in pts)}}}")

# 21 Fano flags = incident (point, line) pairs
fano_flags = []
for li, line in enumerate(fano_lines):
    for pi in sorted(line):
        fano_flags.append((pi, li))

assert len(fano_flags) == 21, f"Expected 21 flags, got {len(fano_flags)}"
print(f"\n  Fano flags (point, line): {len(fano_flags)}")
for fi, (p, l) in enumerate(fano_flags):
    pts_on_line = sorted(fano_lines[l])
    print(f"    Flag {fi:2d}: point {fano_labels[p]} on line L{l} = "
          f"{{{', '.join(fano_labels[q] for q in pts_on_line)}}}")

# Fano permutation action
def fano_perm(elem_idx):
    M = elems[elem_idx]
    perm = []
    for fp in fano_pts:
        image = (M @ fp) % 2
        perm.append(fano_labels.index(''.join(map(str, image.tolist()))))
    return tuple(perm)

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

def fano_fixed_points(perm):
    return [i for i in range(7) if perm[i] == i]

# Action on lines: a line {i,j,k} maps to {perm(i),perm(j),perm(k)}
def line_perm(perm):
    """Return permutation of the 7 lines induced by point permutation."""
    lperm = []
    for line in fano_lines:
        image_line = frozenset(perm[p] for p in line)
        target = fano_lines.index(image_line)
        lperm.append(target)
    return tuple(lperm)

def fixed_lines(lperm):
    return [i for i in range(7) if lperm[i] == i]

# Action on flags: flag (p, l) maps to (perm(p), lperm(l))
def flag_perm(perm, lperm):
    """Return permutation of the 21 flags."""
    fperm = []
    for (p, l) in fano_flags:
        image_flag = (perm[p], lperm[l])
        target = fano_flags.index(image_flag)
        fperm.append(target)
    return tuple(fperm)

def fixed_flags(fperm):
    return [i for i in range(21) if fperm[i] == i]

# =========================================================================
# B_31 CAYLEY SUBGRAPH COMPONENTS
# =========================================================================

# Find PSL generators
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

# Build B_31 Cayley subgraph and find components
b31_list = sorted(B_31)
b31_idx = {e: i for i, e in enumerate(b31_list)}
n_b = len(b31_list)

adj_b = [set() for _ in range(n_b)]
for i, e in enumerate(b31_list):
    for g in psl_gen_set:
        r = mul[e, g]
        if r in b31_idx:
            adj_b[i].add(b31_idx[r])
        l = mul[g, e]
        if l in b31_idx:
            adj_b[i].add(b31_idx[l])

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

# Sort: singletons first, then by size
singletons = sorted([b31_list[list(c)[0]] for c in components if len(c) == 1])
non_singletons = [c for c in components if len(c) > 1]
non_singletons.sort(key=lambda c: (-len(c), min(b31_list[i] for i in c)))

print(f"\n  B_31 Cayley subgraph components: {len(components)}")
print(f"  Singletons: {len(singletons)}")
print(f"  Non-singleton components: {len(non_singletons)} "
      f"(sizes {[len(c) for c in non_singletons]})")

# =========================================================================
# 1. SINGLETON ANALYSIS: FANO ACTION
# =========================================================================
print("\n" + "-" * 72)
print("  1. THE 21 SINGLETONS: FANO FLAG ANALYSIS")
print("-" * 72)

print(f"\n  {'Elem':>5} {'Ord':>4} {'PtCycle':>10} {'LnCycle':>10} "
      f"{'FixPt':>6} {'FixLn':>6} {'FixFlag':>8} {'Flags fixed'}")
print(f"  {'-'*5} {'-'*4} {'-'*10} {'-'*10} {'-'*6} {'-'*6} {'-'*8} {'-'*30}")

singleton_to_flags = {}
flag_to_singleton = {}
all_fixed_flag_counts = []

for e in singletons:
    perm = fano_perm(e)
    pt_cycle = fano_cycle_type(perm)
    lp = line_perm(perm)
    ln_cycle = fano_cycle_type(lp)
    fp = flag_perm(perm, lp)
    fix_pts = fano_fixed_points(perm)
    fix_lns = fixed_lines(lp)
    fix_flgs = fixed_flags(fp)

    singleton_to_flags[e] = fix_flgs
    for fi in fix_flgs:
        if fi not in flag_to_singleton:
            flag_to_singleton[fi] = []
        flag_to_singleton[fi].append(e)

    all_fixed_flag_counts.append(len(fix_flgs))

    flag_desc = [f"({fano_labels[fano_flags[fi][0]]},L{fano_flags[fi][1]})" for fi in fix_flgs]
    print(f"  {e:5d} {ords[e]:4d} {str(pt_cycle):>10} {str(ln_cycle):>10} "
          f"{len(fix_pts):6d} {len(fix_lns):6d} {len(fix_flgs):8d} {' '.join(flag_desc)}")

# Statistics
print(f"\n  Fixed flag counts: {Counter(all_fixed_flag_counts)}")
print(f"  Total singleton-flag incidences: {sum(all_fixed_flag_counts)}")

# Check bijection
flags_hit = set()
for e in singletons:
    flags_hit.update(singleton_to_flags[e])

print(f"\n  Distinct flags fixed by singletons: {len(flags_hit)}/21")
print(f"  All 21 flags hit? {len(flags_hit) == 21}")

# Check: does each singleton fix exactly 1 flag?
exactly_one = all(len(singleton_to_flags[e]) == 1 for e in singletons)
print(f"  Each singleton fixes exactly 1 flag? {exactly_one}")

# Check: is each flag fixed by exactly 1 singleton?
if flags_hit:
    each_flag_once = all(len(flag_to_singleton.get(fi, [])) == 1 for fi in range(21))
    print(f"  Each flag fixed by exactly 1 singleton? {each_flag_once}")

# If not exact bijection, show the actual mapping
if not exactly_one:
    print(f"\n  Detailed flag-fixing pattern:")
    for e in singletons:
        n_flags = len(singleton_to_flags[e])
        if n_flags != 1:
            print(f"    Singleton {e} (order {ords[e]}): fixes {n_flags} flags")

    # Show flag coverage
    print(f"\n  Flag coverage:")
    for fi in range(21):
        p, l = fano_flags[fi]
        fixers = flag_to_singleton.get(fi, [])
        print(f"    Flag {fi:2d} ({fano_labels[p]},L{l}): fixed by {len(fixers)} singletons {fixers}")

# =========================================================================
# 2. NON-SINGLETON ANALYSIS: WHAT ARE THE 10 NON-SINGLETONS?
# =========================================================================
print("\n" + "-" * 72)
print("  2. NON-SINGLETON COMPONENTS: FANO GEOMETRY")
print("-" * 72)

# Identity component
identity_comp = None
for c in non_singletons:
    if any(b31_list[i] == ID for i in c):
        identity_comp = c
        break

# The non-singleton elements
non_sing_elements = []
for c in non_singletons:
    comp_elems = sorted(b31_list[i] for i in c)
    non_sing_elements.extend(comp_elems)

print(f"\n  Non-singleton elements ({len(non_sing_elements)}):")
print(f"  {'Elem':>5} {'Ord':>4} {'Comp':>5} {'PtCycle':>10} {'FixPt':>6} {'FixLn':>6} "
      f"{'FixFlag':>8} {'FixPts':>20} {'FixLns':>15}")
print(f"  {'-'*5} {'-'*4} {'-'*5} {'-'*10} {'-'*6} {'-'*6} {'-'*8} {'-'*20} {'-'*15}")

for ci, c in enumerate(non_singletons):
    comp_elems = sorted(b31_list[i] for i in c)
    for e in comp_elems:
        perm = fano_perm(e)
        pt_cycle = fano_cycle_type(perm)
        lp = line_perm(perm)
        fp = flag_perm(perm, lp)
        fix_pts = fano_fixed_points(perm)
        fix_lns = fixed_lines(lp)
        fix_flgs = fixed_flags(fp)
        fix_pt_labels = [fano_labels[p] for p in fix_pts]
        fix_ln_labels = [f"L{l}" for l in fix_lns]
        print(f"  {e:5d} {ords[e]:4d} {ci+1:5d} {str(pt_cycle):>10} "
              f"{len(fix_pts):6d} {len(fix_lns):6d} {len(fix_flgs):8d} "
              f"{str(fix_pt_labels):>20} {str(fix_ln_labels):>15}")

# =========================================================================
# 3. ORDER-2 ELEMENTS (INVOLUTIONS) AND FANO LINES
# =========================================================================
print("\n" + "-" * 72)
print("  3. INVOLUTIONS IN B_31 AND FANO LINES")
print("-" * 72)

involutions_b31 = sorted(e for e in B_31 if ords[e] == 2)
print(f"\n  B_31 involutions (order 2): {len(involutions_b31)} elements")

print(f"\n  {'Elem':>5} {'Singleton':>10} {'PtCycle':>10} {'FixPt':>6} {'FixLn':>6} "
      f"{'FixPts':>20} {'FixLns':>15}")
print(f"  {'-'*5} {'-'*10} {'-'*10} {'-'*6} {'-'*6} {'-'*20} {'-'*15}")

involution_fixln_map = {}
for e in involutions_b31:
    is_sing = e in singletons
    perm = fano_perm(e)
    pt_cycle = fano_cycle_type(perm)
    lp = line_perm(perm)
    fix_pts = fano_fixed_points(perm)
    fix_lns = fixed_lines(lp)
    fix_pt_labels = [fano_labels[p] for p in fix_pts]
    fix_ln_labels = [f"L{l}" for l in fix_lns]
    involution_fixln_map[e] = fix_lns
    print(f"  {e:5d} {'YES' if is_sing else 'NO':>10} {str(pt_cycle):>10} "
          f"{len(fix_pts):6d} {len(fix_lns):6d} {str(fix_pt_labels):>20} {str(fix_ln_labels):>15}")

# Do the involutions biject to the 7 lines?
all_fixed_lines = set()
for e in involutions_b31:
    all_fixed_lines.update(involution_fixln_map[e])

print(f"\n  Distinct lines fixed by involutions: {len(all_fixed_lines)}/7")
print(f"  All 7 lines covered? {len(all_fixed_lines) == 7}")

# Which involutions fix which lines?
line_fixers = defaultdict(list)
for e in involutions_b31:
    for l in involution_fixln_map[e]:
        line_fixers[l].append(e)

print(f"\n  Line -> fixing involutions:")
for l in range(7):
    pts = sorted(fano_lines[l])
    fixers = line_fixers[l]
    print(f"    L{l} ({{{', '.join(fano_labels[p] for p in pts)}}}): "
          f"fixed by {len(fixers)} involutions: {fixers}")

# =========================================================================
# 4. ORDER-4 ELEMENTS AND FANO STRUCTURE
# =========================================================================
print("\n" + "-" * 72)
print("  4. ORDER-4 ELEMENTS IN B_31")
print("-" * 72)

order4_b31 = sorted(e for e in B_31 if ords[e] == 4)
print(f"\n  B_31 order-4 elements: {len(order4_b31)}")

print(f"\n  {'Elem':>5} {'Singleton':>10} {'PtCycle':>10} {'FixPt':>6} {'FixLn':>6} "
      f"{'Square':>7} {'Sq_ord':>7}")
print(f"  {'-'*5} {'-'*10} {'-'*10} {'-'*6} {'-'*6} {'-'*7} {'-'*7}")

for e in order4_b31:
    is_sing = e in singletons
    perm = fano_perm(e)
    pt_cycle = fano_cycle_type(perm)
    lp = line_perm(perm)
    fix_pts = fano_fixed_points(perm)
    fix_lns = fixed_lines(lp)
    sq = mul[e, e]  # e^2 should be order 2
    sq_ord = ords[sq]
    print(f"  {e:5d} {'YES' if is_sing else 'NO':>10} {str(pt_cycle):>10} "
          f"{len(fix_pts):6d} {len(fix_lns):6d} {sq:7d} {sq_ord:7d}")

# Do the squares of order-4 elements give us the involutions?
squares = set(mul[e, e] for e in order4_b31)
print(f"\n  Squares of order-4 elements: {sorted(squares)}")
print(f"  Are these a subset of involutions? {squares <= set(involutions_b31)}")
print(f"  How many distinct squares? {len(squares)}/{len(order4_b31)}")

# =========================================================================
# 5. ALTERNATIVE: FANO ANTI-FLAGS
# =========================================================================
print("\n" + "-" * 72)
print("  5. ANTI-FLAGS AND OTHER FANO COUNTS")
print("-" * 72)

# Anti-flag = non-incident (point, line) pair: 7*7 - 21 = 28
# But also: 7 points * 4 non-incident lines = 28 anti-flags
n_antiflags = 7 * 7 - 21
print(f"\n  Fano counts:")
print(f"    Points: 7")
print(f"    Lines: 7")
print(f"    Flags (incident point-line): 21")
print(f"    Anti-flags (non-incident point-line): {n_antiflags}")
print(f"    Total (point, line) pairs: {7*7} = 49")

print(f"\n  B_31 counts:")
print(f"    Singletons: {len(singletons)} (vs 21 flags)")
print(f"    Non-singletons: {len(non_sing_elements)} (vs 7+7=14 points+lines, or 10)")
print(f"    Identity: 1")
print(f"    Total: {len(B_31)}")

# 21 = number of elements in PGL(2,F_2) ??? No, |PGL(2,F_2)| = 6
# 21 = number of involutions in PSL(2,7)? Let's check
n_involutions_total = sum(1 for i in range(168) if ords[i] == 2)
print(f"\n  Total involutions in PSL(2,7): {n_involutions_total}")
print(f"  21 = 7*3 = points * points_per_line")
print(f"  21 = C(7,2) = {7*6//2} (pairs of Fano points)")
print(f"  Is 21 = C(7,2)? {21 == 7*6//2}")

# =========================================================================
# 6. SINGLETON GROUPING BY FANO STRUCTURE
# =========================================================================
print("\n" + "-" * 72)
print("  6. SINGLETON GROUPING BY FANO ACTION TYPE")
print("-" * 72)

# Group singletons by (order, point_cycle_type, line_cycle_type)
singleton_types = defaultdict(list)
for e in singletons:
    perm = fano_perm(e)
    pt_cycle = fano_cycle_type(perm)
    lp = line_perm(perm)
    ln_cycle = fano_cycle_type(lp)
    key = (int(ords[e]), pt_cycle, ln_cycle)
    singleton_types[key].append(e)

print(f"\n  Singleton types (order, pt_cycle, ln_cycle):")
for key, elems_list in sorted(singleton_types.items()):
    print(f"    {key}: {len(elems_list)} elements {elems_list}")

# =========================================================================
# 7. POINT-LINE DUALITY: DO SINGLETONS RESPECT IT?
# =========================================================================
print("\n" + "-" * 72)
print("  7. POINT-LINE DUALITY")
print("-" * 72)

# In PG(2,2), there is a duality swapping points and lines.
# Under duality, a flag (p, L) maps to (L*, p*) where * is the dual.
# Do the singletons pair up under this duality?

# The duality in F_2^3 can be implemented as the transpose-inverse action
# (or equivalently, the action on the dual space)

# For each singleton, compute its action on both points and lines
# and check if there's a "dual" singleton
print(f"\n  Checking if singletons pair under point-line duality...")

# The duality: if g acts on points as perm P, it acts on lines as perm L.
# The dual element g* would act on points as L and on lines as P.
# Does such a g* exist among singletons?

singleton_set = set(singletons)
dual_pairs = []
dual_visited = set()

for e in singletons:
    if e in dual_visited:
        continue
    perm_e = fano_perm(e)
    lp_e = line_perm(perm_e)

    # Find singleton whose point-perm = e's line-perm
    # (in terms of cycle types, at minimum)
    e_pt_cycle = fano_cycle_type(perm_e)
    e_ln_cycle = fano_cycle_type(lp_e)

    dual_found = None
    for f in singletons:
        if f == e or f in dual_visited:
            continue
        perm_f = fano_perm(f)
        lp_f = line_perm(perm_f)
        f_pt_cycle = fano_cycle_type(perm_f)
        f_ln_cycle = fano_cycle_type(lp_f)

        # Weak duality: cycle types swap
        if f_pt_cycle == e_ln_cycle and f_ln_cycle == e_pt_cycle and e_pt_cycle != e_ln_cycle:
            dual_found = f
            break

    if dual_found is not None:
        dual_pairs.append((e, dual_found))
        dual_visited.add(e)
        dual_visited.add(dual_found)
    else:
        # Self-dual?
        if e_pt_cycle == e_ln_cycle:
            dual_pairs.append((e, e))
            dual_visited.add(e)

print(f"  Dual pairs found: {len(dual_pairs)}")
for a, b in dual_pairs[:10]:
    if a == b:
        print(f"    {a} (self-dual)")
    else:
        print(f"    {a} <-> {b}")

unpaired = [e for e in singletons if e not in dual_visited]
print(f"  Unpaired singletons: {len(unpaired)}")

# =========================================================================
# 8. THE 21 INVOLUTIONS OF PSL(2,7) AND THE SINGLETONS
# =========================================================================
print("\n" + "-" * 72)
print("  8. THE 21 INVOLUTIONS AND THE 21 SINGLETONS")
print("-" * 72)

all_involutions = sorted(i for i in range(168) if ords[i] == 2)
print(f"\n  Total involutions in PSL(2,7): {len(all_involutions)}")
print(f"  Involutions in B_31: {len(involutions_b31)}")
print(f"  Singletons that are involutions: {len([e for e in singletons if ords[e] == 2])}")
print(f"  Singletons that are order-4: {len([e for e in singletons if ords[e] == 4])}")

# Are the 21 singletons exactly the 21 involutions?
sing_set = set(singletons)
inv_set = set(all_involutions)
print(f"\n  Singletons = involutions? {sing_set == inv_set}")
print(f"  |Singletons & involutions| = {len(sing_set & inv_set)}")
print(f"  Singletons not involutions: {sorted(sing_set - inv_set)}")
print(f"  Involutions not singletons: {sorted(inv_set - sing_set)}")

# Involutions in each stratum
inv_in_b = sorted(inv_set & B_31)
inv_in_z = sorted(inv_set & Z_62)
inv_in_t = sorted(inv_set & T_75)
print(f"\n  Involutions by stratum: B={len(inv_in_b)}, Z={len(inv_in_z)}, T={len(inv_in_t)}")

# =========================================================================
# SUMMARY
# =========================================================================
print("\n" + "=" * 72)
print("  SUMMARY: FANO FLAGS AND MATTER")
print("=" * 72)

n_fix_1 = sum(1 for e in singletons if len(singleton_to_flags[e]) == 1)
n_fix_0 = sum(1 for e in singletons if len(singleton_to_flags[e]) == 0)
n_fix_many = sum(1 for e in singletons if len(singleton_to_flags[e]) > 1)

print(f"""
  B_31 COMPONENT STRUCTURE (Cayley subgraph):
    21 singletons + 2 triples + 2 doubles = 25 components = 31 elements

  FANO FLAG TEST:
    Singletons fixing exactly 1 flag: {n_fix_1}/21
    Singletons fixing 0 flags: {n_fix_0}/21
    Singletons fixing >1 flag: {n_fix_many}/21
    Distinct flags covered: {len(flags_hit)}/21
    BIJECTION? {n_fix_1 == 21 and len(flags_hit) == 21}

  INVOLUTION CONNECTION:
    Total involutions in PSL(2,7): {len(all_involutions)}
    21 singletons = 21 involutions? {sing_set == inv_set}
    Overlap: {len(sing_set & inv_set)}

  VERDICT:
    {'The 21 singletons ARE the 21 Fano flags.' if n_fix_1 == 21 and len(flags_hit) == 21 else 'The flag bijection does NOT hold in its simple form. See detailed analysis.'}
""")
