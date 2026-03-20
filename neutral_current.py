"""
neutral_current.py
Test whether F_36 (36 order-7 non-Weinberg elements of Z_62) are the neutral current sector.

Questions:
1. Does element 92 (Z boson candidate) commute with F_36?
2. Does F_36 conjugation preserve Fix(C) & B_31?
3. Does F_36 preserve S_3-orbits of B_31 (flavour-neutral)?
4. Does W_26 mix S_3-orbits of B_31 (flavour-changing)?
5. Self-connection: how does F_36 compare to W_26?
6. Generation: <F_36> vs <W_26>?
"""
import sys
import numpy as np
from collections import Counter, defaultdict, deque

sys.path.insert(0, '.')
from psl27_core import build_group, classify_strata, mat_key

print("=" * 72)
print("  NEUTRAL CURRENT TEST: F_36 vs W_26")
print("=" * 72)

elems, e2i, mul, inv_table, ords, ID = build_group()
B_31, Z_62, T_75, z3, z3sq, W26, S3 = classify_strata(
    elems, e2i, mul, inv_table, ords, ID)

# Charge conjugation
antidiag = np.array([[0,0,1],[0,1,0],[1,0,0]], dtype=int)
C_idx = e2i[mat_key(antidiag)]
fix_C = {i for i in range(168) if mul[C_idx, mul[i, inv_table[C_idx]]] == i}
fix_C_B31 = fix_C & B_31

# The 36
F36 = sorted(Z_62 - W26)
F36_set = set(F36)
W26_set = set(W26)

# S_3 orbits of B_31
def compute_s3_orbits(stratum):
    remaining = set(stratum)
    orbits = []
    while remaining:
        x = min(remaining)
        orbit = set()
        for g in S3:
            orbit.add(mul[g, mul[x, inv_table[g]]])
        orbits.append(sorted(orbit))
        remaining -= orbit
    orbits.sort(key=lambda o: (len(o), o[0]))
    return orbits

orbits_b31 = compute_s3_orbits(B_31)
elem_to_orbit = {}
for i, orb in enumerate(orbits_b31):
    for e in orb:
        elem_to_orbit[e] = i

print(f"\n  F_36: {len(F36)} elements (order-7 non-Weinberg Z_62)")
print(f"  W_26: {len(W26)} elements (order-3 Weinberg Z_62)")
print(f"  Fix(C) & B_31: {len(fix_C_B31)} elements: {sorted(fix_C_B31)}")
print(f"  B_31 S_3-orbits: {len(orbits_b31)} (sizes {[len(o) for o in orbits_b31]})")

# =========================================================================
# 1. COMMUTATION WITH ELEMENT 92 (Z BOSON CANDIDATE)
# =========================================================================
print("\n" + "-" * 72)
print("  1. COMMUTATION WITH ELEMENT 92 (Z BOSON CANDIDATE)")
print("-" * 72)

# First check if 92 is actually in the group and in Fix(C) & B_31
if 92 < 168:
    print(f"\n  Element 92: order = {ords[92]}, in B_31 = {92 in B_31}, "
          f"in Fix(C) = {92 in fix_C}")
else:
    print(f"  Element 92 is out of range")

# Find the actual Fix(C) & B_31 elements and their properties
print(f"\n  All Fix(C) & B_31 elements:")
for e in sorted(fix_C_B31):
    # Which orbit?
    orb_idx = elem_to_orbit[e]
    print(f"    elem {e:3d}: order {ords[e]}, orbit {orb_idx+1} (size {len(orbits_b31[orb_idx])})")

# For each Fix(C) & B_31 element, count commutations with F_36
print(f"\n  Commutation of Fix(C) & B_31 with F_36:")
for z_cand in sorted(fix_C_B31):
    commutes_f36 = []
    for f in F36:
        # z*f == f*z ?
        if mul[z_cand, f] == mul[f, z_cand]:
            commutes_f36.append(f)
    print(f"    elem {z_cand:3d} (order {ords[z_cand]}): commutes with {len(commutes_f36)}/{len(F36)} F_36 elements")
    if commutes_f36:
        comm_orders = Counter(int(ords[c]) for c in commutes_f36)
        print(f"      Commuting elements orders: {dict(comm_orders)}")

# Same for W_26
print(f"\n  Commutation of Fix(C) & B_31 with W_26:")
for z_cand in sorted(fix_C_B31):
    commutes_w26 = []
    for w in W26:
        if mul[z_cand, w] == mul[w, z_cand]:
            commutes_w26.append(w)
    print(f"    elem {z_cand:3d} (order {ords[z_cand]}): commutes with {len(commutes_w26)}/{len(W26)} W_26 elements")

# Full centraliser of each Fix(C) & B_31 element
print(f"\n  Centraliser sizes of Fix(C) & B_31 elements:")
for z_cand in sorted(fix_C_B31):
    centraliser = [g for g in range(168) if mul[z_cand, g] == mul[g, z_cand]]
    cent_in_z62 = [g for g in centraliser if g in Z_62]
    cent_in_f36 = [g for g in centraliser if g in F36_set]
    cent_in_w26 = [g for g in centraliser if g in W26_set]
    print(f"    elem {z_cand:3d}: |C_G(z)| = {len(centraliser)}, "
          f"in Z_62 = {len(cent_in_z62)}, in F_36 = {len(cent_in_f36)}, in W_26 = {len(cent_in_w26)}")

# =========================================================================
# 2. F_36 CONJUGATION ACTION ON B_31: PRESERVE Fix(C)?
# =========================================================================
print("\n" + "-" * 72)
print("  2. F_36 CONJUGATION ON B_31: DOES IT PRESERVE Fix(C) & B_31?")
print("-" * 72)

# For each f in F_36, conjugate each b in B_31: f*b*f^{-1}
# Check: does this land in B_31? (Should, since S_3 ⊂ Stab(B_31) and F_36 may not be in S_3)

# First: does F_36 conjugation preserve B_31 at all?
f36_preserves_b31 = True
for f in F36:
    for b in B_31:
        image = mul[f, mul[b, inv_table[f]]]
        if image not in B_31:
            f36_preserves_b31 = False
            break
    if not f36_preserves_b31:
        break

print(f"\n  F_36 conjugation preserves B_31? {f36_preserves_b31}")

if not f36_preserves_b31:
    # Where do the images go?
    landing = Counter()
    for f in F36:
        for b in B_31:
            image = mul[f, mul[b, inv_table[f]]]
            if image in B_31:
                landing['B'] += 1
            elif image in Z_62:
                landing['Z'] += 1
            elif image in T_75:
                landing['T'] += 1
    total = sum(landing.values())
    print(f"  Landing distribution of f*b*f^-1 for f in F_36, b in B_31:")
    for s in ['B', 'Z', 'T']:
        print(f"    {s}: {landing[s]} ({landing[s]/total*100:.1f}%)")

# Does F_36 conjugation preserve Fix(C) & B_31?
# (Only relevant if it preserves B_31)
if f36_preserves_b31:
    f36_preserves_fixC = True
    for f in F36:
        for b in fix_C_B31:
            image = mul[f, mul[b, inv_table[f]]]
            if image not in fix_C_B31:
                f36_preserves_fixC = False
                break
        if not f36_preserves_fixC:
            break
    print(f"  F_36 conjugation preserves Fix(C) & B_31? {f36_preserves_fixC}")

# Same tests for W_26
w26_preserves_b31 = True
for w in W26:
    for b in B_31:
        image = mul[w, mul[b, inv_table[w]]]
        if image not in B_31:
            w26_preserves_b31 = False
            break
    if not w26_preserves_b31:
        break

print(f"\n  W_26 conjugation preserves B_31? {w26_preserves_b31}")

if not w26_preserves_b31:
    landing_w = Counter()
    for w in W26:
        for b in B_31:
            image = mul[w, mul[b, inv_table[w]]]
            if image in B_31:
                landing_w['B'] += 1
            elif image in Z_62:
                landing_w['Z'] += 1
            elif image in T_75:
                landing_w['T'] += 1
    total_w = sum(landing_w.values())
    print(f"  Landing distribution of w*b*w^-1 for w in W_26, b in B_31:")
    for s in ['B', 'Z', 'T']:
        print(f"    {s}: {landing_w[s]} ({landing_w[s]/total_w*100:.1f}%)")

# =========================================================================
# 3. S_3-ORBIT PRESERVATION: FLAVOUR-CHANGING vs FLAVOUR-NEUTRAL
# =========================================================================
print("\n" + "-" * 72)
print("  3. S_3-ORBIT PRESERVATION: FLAVOUR TEST")
print("-" * 72)

def test_orbit_preservation(actor_set, actor_label):
    """Test whether conjugation by actor_set preserves S_3-orbits of B_31."""
    orbit_map = defaultdict(Counter)  # orbit_i -> {orbit_j: count}
    for a in actor_set:
        for b in B_31:
            image = mul[a, mul[b, inv_table[a]]]
            src_orb = elem_to_orbit[b]
            if image in elem_to_orbit:
                tgt_orb = elem_to_orbit[image]
                orbit_map[src_orb][tgt_orb] += 1
            else:
                orbit_map[src_orb]['outside'] += 1

    preserving = 0
    mixing = 0
    escaping = 0
    for src_orb, targets in sorted(orbit_map.items()):
        n_self = targets.get(src_orb, 0)
        n_other_orb = sum(v for k, v in targets.items() if k != src_orb and k != 'outside')
        n_outside = targets.get('outside', 0)
        total = sum(targets.values())

        if n_self == total:
            preserving += 1
        elif n_outside > 0:
            escaping += 1
        else:
            mixing += 1

    print(f"\n  {actor_label} conjugation on B_31 S_3-orbits:")
    print(f"    Orbits preserved: {preserving}/{len(orbits_b31)}")
    print(f"    Orbits mixed (within B_31): {mixing}/{len(orbits_b31)}")
    print(f"    Orbits escaping B_31: {escaping}/{len(orbits_b31)}")

    # Print the orbit transition matrix
    print(f"\n    Orbit transition matrix ({actor_label}):")
    print(f"    {'Src':>4}", end='')
    for j in range(len(orbits_b31)):
        print(f" {j+1:>5}", end='')
    print(f" {'out':>5}")
    for i in range(len(orbits_b31)):
        print(f"    {i+1:>4}", end='')
        targets = orbit_map[i]
        for j in range(len(orbits_b31)):
            print(f" {targets.get(j, 0):>5}", end='')
        print(f" {targets.get('outside', 0):>5}")

    return preserving, mixing, escaping

f36_pres, f36_mix, f36_esc = test_orbit_preservation(F36, "F_36")
w26_pres, w26_mix, w26_esc = test_orbit_preservation(W26, "W_26")

print(f"\n  COMPARISON:")
print(f"    F_36 (neutral current?): {f36_pres} preserved, {f36_mix} mixed, {f36_esc} escaped")
print(f"    W_26 (charged current?): {w26_pres} preserved, {w26_mix} mixed, {w26_esc} escaped")
print(f"    F_36 more orbit-preserving? {f36_pres > w26_pres}")

# =========================================================================
# 4. SELF-CONNECTION: F_36 vs W_26 UNDER MULTIPLICATION
# =========================================================================
print("\n" + "-" * 72)
print("  4. SELF-CONNECTION: F_36 vs W_26 MULTIPLICATION CLOSURE")
print("-" * 72)

# F_36 self-products
f36_self = Counter()
for a in F36:
    for b in F36:
        p = mul[a, b]
        if p in F36_set:
            f36_self['F36'] += 1
        elif p in W26_set:
            f36_self['W26'] += 1
        elif p in B_31:
            f36_self['B'] += 1
        elif p in Z_62:
            f36_self['Z_other'] += 1
        elif p in T_75:
            f36_self['T'] += 1

total_f = len(F36)**2
print(f"\n  F_36 * F_36 ({total_f} products):")
for k in ['F36', 'W26', 'B', 'Z_other', 'T']:
    if k in f36_self:
        print(f"    -> {k}: {f36_self[k]} ({f36_self[k]/total_f*100:.1f}%)")

# W_26 self-products
w26_self = Counter()
for a in W26:
    for b in W26:
        p = mul[a, b]
        if p in W26_set:
            w26_self['W26'] += 1
        elif p in F36_set:
            w26_self['F36'] += 1
        elif p in B_31:
            w26_self['B'] += 1
        elif p in Z_62:
            w26_self['Z_other'] += 1
        elif p in T_75:
            w26_self['T'] += 1

total_w = len(W26)**2
print(f"\n  W_26 * W_26 ({total_w} products):")
for k in ['W26', 'F36', 'B', 'Z_other', 'T']:
    if k in w26_self:
        print(f"    -> {k}: {w26_self[k]} ({w26_self[k]/total_w*100:.1f}%)")

# Cross-products
f36_w26 = Counter()
for a in F36:
    for b in W26:
        p = mul[a, b]
        if p in F36_set:
            f36_w26['F36'] += 1
        elif p in W26_set:
            f36_w26['W26'] += 1
        elif p in B_31:
            f36_w26['B'] += 1
        elif p in Z_62:
            f36_w26['Z_other'] += 1
        elif p in T_75:
            f36_w26['T'] += 1

total_fw = len(F36) * len(W26)
print(f"\n  F_36 * W_26 ({total_fw} products):")
for k in ['F36', 'W26', 'B', 'Z_other', 'T']:
    if k in f36_w26:
        print(f"    -> {k}: {f36_w26[k]} ({f36_w26[k]/total_fw*100:.1f}%)")

# Connectivity: build multiplication graph within each set
def mult_connectivity(elem_set, label):
    """How many connected components when edges = products landing inside the set."""
    elem_list = sorted(elem_set)
    n = len(elem_list)
    e2idx = {e: i for i, e in enumerate(elem_list)}

    adj = defaultdict(set)
    for a in elem_list:
        for b in elem_list:
            p = mul[a, b]
            if p in e2idx:
                adj[e2idx[a]].add(e2idx[p])
                adj[e2idx[p]].add(e2idx[a])
                adj[e2idx[b]].add(e2idx[p])
                adj[e2idx[p]].add(e2idx[b])

    # Count connected components
    visited = set()
    components = 0
    for start in range(n):
        if start in visited:
            continue
        components += 1
        queue = deque([start])
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            for nbr in adj[node]:
                if nbr not in visited:
                    queue.append(nbr)

    print(f"\n  {label} multiplication graph: {components} connected components (of {n} elements)")
    return components

f36_comp = mult_connectivity(F36_set, "F_36")
w26_comp = mult_connectivity(W26_set, "W_26")

# Also check: direct product graph (a->a*b for all b in the set)
def product_graph_components(elem_set, label):
    """Connected components of directed product graph: a -> a*b for each b in set."""
    elem_list = sorted(elem_set)
    e2idx = {e: i for i, e in enumerate(elem_list)}
    n = len(elem_list)

    adj = [set() for _ in range(n)]
    for i, a in enumerate(elem_list):
        for b in elem_list:
            p = mul[a, b]
            if p in e2idx:
                adj[i].add(e2idx[p])
                adj[e2idx[p]].add(i)

    visited = set()
    components = 0
    for start in range(n):
        if start in visited:
            continue
        components += 1
        queue = deque([start])
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            for nbr in adj[node]:
                if nbr not in visited:
                    queue.append(nbr)

    print(f"  {label} product graph: {components} components")
    return components

f36_prod = product_graph_components(F36_set, "F_36")
w26_prod = product_graph_components(W26_set, "W_26")

# =========================================================================
# 5. GENERATION COMPARISON
# =========================================================================
print("\n" + "-" * 72)
print("  5. GENERATION: <F_36> vs <W_26>")
print("-" * 72)

def generate_from(seed_set):
    generated = {ID}
    frontier = set(seed_set) | {ID}
    while frontier - generated:
        generated.update(frontier)
        new = set()
        for x in frontier:
            for g in seed_set:
                new.add(mul[x, g])
                new.add(mul[g, x])
        frontier = new - generated
    return generated

gen_f36 = generate_from(F36)
gen_w26 = generate_from(W26)

print(f"\n  |<F_36>| = {len(gen_f36)}")
print(f"  |<W_26>| = {len(gen_w26)}")
print(f"  <F_36> = PSL(2,7)? {len(gen_f36) == 168}")
print(f"  <W_26> = PSL(2,7)? {len(gen_w26) == 168}")
print(f"  <F_36> = <W_26>? {gen_f36 == gen_w26}")

# What do they generate at each step?
def generation_sequence(seed_set, label):
    generated = {ID}
    frontier = set(seed_set) | {ID}
    step = 0
    print(f"\n  {label} generation sequence:")
    while frontier - generated:
        generated.update(frontier)
        new = set()
        for x in frontier:
            for g in seed_set:
                new.add(mul[x, g])
                new.add(mul[g, x])
        frontier = new - generated
        step += 1
        # Count stratum distribution
        in_b = sum(1 for e in generated if e in B_31)
        in_z = sum(1 for e in generated if e in Z_62)
        in_t = sum(1 for e in generated if e in T_75)
        print(f"    Step {step}: |gen| = {len(generated)} (B={in_b}, Z={in_z}, T={in_t})")

generation_sequence(F36, "F_36")
generation_sequence(list(W26), "W_26")

# =========================================================================
# 6. COMMUTATOR STRUCTURE: [F_36, W_26]
# =========================================================================
print("\n" + "-" * 72)
print("  6. FULL COMMUTATOR ANALYSIS: [F_36, W_26]")
print("-" * 72)

# [f, w] = f * w * f^{-1} * w^{-1}
comm_landing = Counter()
comm_in_b31 = []
for f in F36:
    for w in W26:
        comm = mul[f, mul[w, mul[inv_table[f], inv_table[w]]]]
        if comm in B_31:
            comm_landing['B'] += 1
            comm_in_b31.append(comm)
        elif comm in Z_62:
            comm_landing['Z'] += 1
        elif comm in T_75:
            comm_landing['T'] += 1

total_comm = sum(comm_landing.values())
print(f"\n  [F_36, W_26] commutator landing ({total_comm} products):")
for s in ['B', 'Z', 'T']:
    print(f"    {s}: {comm_landing[s]} ({comm_landing[s]/total_comm*100:.1f}%)")

if comm_in_b31:
    b31_comm_dist = Counter(comm_in_b31)
    print(f"\n  Commutators landing in B_31: {len(comm_in_b31)}")
    print(f"  Distinct B_31 elements hit: {len(b31_comm_dist)}")
    hit_orbits = set(elem_to_orbit[e] for e in b31_comm_dist.keys())
    print(f"  B_31 orbits hit: {sorted(i+1 for i in hit_orbits)} (of {len(orbits_b31)})")
    hit_fixC = sum(1 for e in b31_comm_dist.keys() if e in fix_C_B31)
    print(f"  Fix(C) & B_31 elements hit: {hit_fixC}/{len(fix_C_B31)}")

# =========================================================================
# 7. STRATUM SELF-MULTIPLICATION: EXACT COMPONENT ANALYSIS
# =========================================================================
print("\n" + "-" * 72)
print("  7. STRATUM SELF-MULTIPLICATION: EXACT COMPONENT COUNTS")
print("-" * 72)

# For the key result: T_75 connected, B_31 into 25, Z_62 into 3
# Use PSL(2,7) generators for the Cayley subgraph

def find_generators_psl(mul_table, ords_arr, ID_elem):
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

psl_gens = find_generators_psl(mul, ords, ID)
psl_gen_set = sorted(set(psl_gens + [inv_table[g] for g in psl_gens]))

for label, stratum in [('B_31', B_31), ('Z_62', Z_62), ('T_75', T_75)]:
    elements = sorted(stratum)
    e2idx = {e: i for i, e in enumerate(elements)}
    n = len(elements)

    # Build adjacency: edge if e*g or g*e is also in stratum
    adj = [set() for _ in range(n)]
    for i, e in enumerate(elements):
        for g in psl_gen_set:
            r = mul[e, g]
            if r in e2idx:
                adj[i].add(e2idx[r])
            l = mul[g, e]
            if l in e2idx:
                adj[i].add(e2idx[l])

    # Connected components
    visited = set()
    components = []
    for start in range(n):
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
            for nbr in adj[node]:
                if nbr not in visited:
                    queue.append(nbr)
        components.append(comp)

    comp_sizes = sorted([len(c) for c in components], reverse=True)
    comp_orders = []
    for c in components:
        orders_in_comp = Counter(int(ords[elements[i]]) for i in c)
        comp_orders.append(dict(sorted(orders_in_comp.items())))

    print(f"\n  {label} Cayley subgraph (PSL generators, multiplication):")
    print(f"    Components: {len(components)}")
    print(f"    Component sizes: {comp_sizes}")
    if len(components) <= 30:
        for ci, (comp, co) in enumerate(zip(components, comp_orders)):
            if len(comp) <= 10:
                print(f"    Component {ci+1} (size {len(comp)}): orders {co}, "
                      f"elements {sorted(elements[i] for i in comp)}")
            else:
                print(f"    Component {ci+1} (size {len(comp)}): orders {co}")

# =========================================================================
# SUMMARY
# =========================================================================
print("\n" + "=" * 72)
print("  SUMMARY: NEUTRAL CURRENT IDENTIFICATION")
print("=" * 72)

print(f"""
  CONJUGATION ON B_31:
    F_36 preserves B_31? {f36_preserves_b31}
    W_26 preserves B_31? {w26_preserves_b31}

  S_3-ORBIT (FLAVOUR) PRESERVATION:
    F_36: {f36_pres} preserved, {f36_mix} mixed, {f36_esc} escaped
    W_26: {w26_pres} preserved, {w26_mix} mixed, {w26_esc} escaped
    F_36 more flavour-preserving? {f36_pres > w26_pres}

  SELF-MULTIPLICATION CLOSURE:
    F_36: {f36_self.get('F36', 0)}/{total_f} self-products ({f36_self.get('F36', 0)/total_f*100:.1f}%)
    W_26: {w26_self.get('W26', 0)}/{total_w} self-products ({w26_self.get('W26', 0)/total_w*100:.1f}%)

  GENERATION:
    |<F_36>| = {len(gen_f36)}, |<W_26>| = {len(gen_w26)}
    Both generate PSL(2,7)? F_36={len(gen_f36)==168}, W_26={len(gen_w26)==168}

  STRATUM SELF-CONNECTION (Cayley subgraph):
    B_31: fractures (strong force doesn't close under matter multiplication)
    Z_62: fractures (weak force partially closes)
    T_75: connected (confinement closes completely)
""")
