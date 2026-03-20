"""
resolve_inconsistency.py
Run the FULL commutator + stability analysis using the SAME generator pair
as fano_flags_matter.py: PSL generators (32, 1) giving 21 singletons = 12 flags + 9 lines.

This resolves the critical inconsistency between the paper narrative (9 lines)
and gauge_boson_ratio.py tables (8 lines).
"""
import sys, io, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, '.')
from psl27_core import build_group, classify_strata, conjugacy_classes
from collections import Counter, defaultdict, deque
from itertools import product as iproduct
from math import gcd

elems, e2i, mul, inv_table, ords, ID = build_group()
B_31, Z_62, T_75, z3, z3sq, W26, S3 = classify_strata(elems, e2i, mul, inv_table, ords, ID)
F36 = Z_62 - W26
b31_set = set(B_31)

print("=" * 70)
print("  RESOLVING INCONSISTENCY: 9 LINE INVOLUTIONS vs 8")
print("=" * 70)

# ============================================================
# STEP 1: Build singletons with gen_set [1, 32, 102]
# ============================================================
psl_gen_set = sorted(set([1, 32, inv_table[1], inv_table[32]]))
print(f"\n  PSL gen_set (fano_flags_matter): {psl_gen_set}")

b31_list = sorted(B_31)
adj = {e: set() for e in b31_list}
for e in b31_list:
    for g in psl_gen_set:
        r = mul[e, g]
        if r in b31_set:
            adj[e].add(r)
        l = mul[g, e]
        if l in b31_set:
            adj[e].add(l)

visited = set()
singletons = []
for start in b31_list:
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
    if len(comp) == 1:
        singletons.append(start)

flag12 = sorted([e for e in singletons if ords[e] == 4])
line9 = sorted([e for e in singletons if ords[e] == 2])
print(f"  Singletons: {len(singletons)} = {len(flag12)} flags + {len(line9)} lines")
print(f"  flag12: {flag12}")
print(f"  line9:  {line9}")

# ============================================================
# STEP 2: Commutator analysis with 9 line involutions
# ============================================================
print("\n" + "=" * 70)
print("  COMMUTATOR ANALYSIS: 9 LINE INVOLUTIONS")
print("=" * 70)

comm_landing = Counter()
comm_elements = defaultdict(list)

for a in line9:
    for b in line9:
        if a == b:
            continue
        c = mul[a, mul[b, mul[a, b]]]  # [a,b] = abab for involutions

        if c == ID:
            comm_landing["ID"] += 1
        elif c in W26:
            comm_landing["W26"] += 1
            comm_elements["W26"].append(c)
        elif c in F36:
            comm_landing["F36"] += 1
            comm_elements["F36"].append(c)
        elif c in b31_set:
            comm_landing["B31"] += 1
            comm_elements["B31"].append(c)
        elif c in T_75:
            comm_landing["T75"] += 1
            comm_elements["T75"].append(c)

total = sum(comm_landing.values())
print(f"\n  Total off-diagonal commutators: {total} (expected 9*8={9*8})")
print(f"\n  Landing distribution:")
for s in ["ID", "W26", "F36", "B31", "T75"]:
    n = comm_landing.get(s, 0)
    print(f"    {s}: {n} ({n/total*100:.1f}%)")

# Ratio test
w = comm_landing.get("W26", 0)
t = comm_landing.get("T75", 0)
if t > 0:
    g = gcd(w, t)
    print(f"\n  W26 : T75 = {w} : {t} = {w//g} : {t//g}")
    print(f"  Target 3:8 = {3/8:.6f}, actual = {w/t:.6f}")
    print(f"  MATCH 3:8? {w//g == 3 and t//g == 8}")

# Distinct elements
for sector in ["W26", "F36", "B31", "T75"]:
    if sector in comm_elements:
        distinct = sorted(set(comm_elements[sector]))
        print(f"\n  Distinct {sector} elements: {len(distinct)}")
        for e in distinct:
            count = comm_elements[sector].count(e)
            print(f"    elem {e}: order {ords[e]}, appears {count}x")

# ============================================================
# STEP 3: T75 orbit coverage
# ============================================================
print("\n" + "-" * 70)
print("  T75 ORBIT COVERAGE")
print("-" * 70)

remaining_t = set(T_75)
orbits_t75 = []
while remaining_t:
    x = min(remaining_t)
    orbit = set()
    for g_elem in S3:
        orbit.add(mul[g_elem, mul[x, inv_table[g_elem]]])
    orbits_t75.append(sorted(orbit))
    remaining_t -= orbit
orbits_t75.sort(key=lambda o: (len(o), o[0]))

t75_comm_set = set(comm_elements.get("T75", []))
orbits_hit = set()
for e in t75_comm_set:
    for oi, orb in enumerate(orbits_t75):
        if e in orb:
            orbits_hit.add(oi)
            break

print(f"\n  T75 orbits hit: {len(orbits_hit)}/{len(orbits_t75)}")
print(f"  Target dim(SU(3)) = 8")
print(f"  MATCH? {len(orbits_hit) == 8}")

for oi, orb in enumerate(orbits_t75):
    hit = any(e in t75_comm_set for e in orb)
    n_hit = sum(1 for e in orb if e in t75_comm_set)
    orb_ords = sorted(set(int(ords[e]) for e in orb))
    marker = "HIT" if hit else "---"
    print(f"    Orbit {oi+1:2d} (size {len(orb)}, orders {orb_ords}): {marker} ({n_hit} elements)")

# ============================================================
# STEP 4: Stability criterion with 12 flag elements
# ============================================================
print("\n" + "=" * 70)
print("  STABILITY CRITERION: 12 FLAG ELEMENTS")
print("=" * 70)

fano_pts = []
fano_labels = []
for bits in iproduct([0, 1], repeat=3):
    if any(b for b in bits):
        fano_pts.append(np.array(bits, dtype=int))
        fano_labels.append("".join(map(str, bits)))

fano_lines = []
for i in range(7):
    for j in range(i + 1, 7):
        k_vec = (fano_pts[i] + fano_pts[j]) % 2
        k_label = "".join(map(str, k_vec.tolist()))
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
        perm.append(fano_labels.index("".join(map(str, image.tolist()))))
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


def fixed_lines_fn(lperm):
    return [i for i in range(7) if lperm[i] == i]


def get_fixed_flag(e):
    perm = fano_perm(e)
    lp = line_perm(perm)
    for pi in fano_fixed_points(perm):
        for li in fixed_lines_fn(lp):
            if pi in fano_lines[li]:
                return (pi, li)
    return None


# Find Cartan line
s3_common_lines = None
for g_elem in S3:
    perm = fano_perm(g_elem)
    lp = line_perm(perm)
    fl = set(fixed_lines_fn(lp))
    s3_common_lines = fl if s3_common_lines is None else s3_common_lines & fl
cartan_line = min(s3_common_lines)
cartan_pts = sorted(fano_lines[cartan_line])
cartan_labels = [fano_labels[p] for p in cartan_pts]
print(f"\n  Cartan line: L{cartan_line} = {{{', '.join(cartan_labels)}}}")

stable = [e for e in flag12 if mul[e, e] in b31_set]
unstable = [e for e in flag12 if mul[e, e] in T_75]

print(f"  Stable flags: {len(stable)}, Unstable flags: {len(unstable)}")
print(f"\n  {'Elem':>5} {'FixPt':>8} {'FixLn':>6} {'On_L1':>6} {'Square':>7} {'Sq_str':>7} {'Verdict':>8}")
print(f"  {'-'*5} {'-'*8} {'-'*6} {'-'*6} {'-'*7} {'-'*7} {'-'*8}")

for e in flag12:
    flag = get_fixed_flag(e)
    if flag is None:
        print(f"  {e:5d}  NO FLAG")
        continue
    p, l = flag
    sq = mul[e, e]
    is_stable = sq in b31_set
    on_cartan = p in fano_lines[cartan_line]
    sq_str = "B" if sq in b31_set else ("T" if sq in T_75 else "Z")
    verdict = "STABLE" if is_stable else "UNSTABLE"
    print(f"  {e:5d} {fano_labels[p]:>8} {'L'+str(l):>6} {'ON' if on_cartan else 'OFF':>6} "
          f"{sq:7d} {sq_str:>7} {verdict:>8}")

stable_on = sum(1 for e in stable if get_fixed_flag(e) and get_fixed_flag(e)[0] in fano_lines[cartan_line])
unstable_on = sum(1 for e in unstable if get_fixed_flag(e) and get_fixed_flag(e)[0] in fano_lines[cartan_line])
print(f"\n  Stable ON Cartan: {stable_on}/{len(stable)}")
print(f"  Stable OFF Cartan: {len(stable) - stable_on}/{len(stable)}")
print(f"  Unstable ON Cartan: {unstable_on}/{len(unstable)}")
print(f"  Unstable OFF Cartan: {len(unstable) - unstable_on}/{len(unstable)}")
criterion = (stable_on == 0 and unstable_on == len(unstable))
print(f"\n  CRITERION HOLDS: {criterion}")

# ============================================================
# STEP 5: Line classification
# ============================================================
print("\n" + "-" * 70)
print("  LINE CLASSIFICATION")
print("-" * 70)
for li in range(7):
    s_count = sum(1 for e in stable if get_fixed_flag(e) and get_fixed_flag(e)[1] == li)
    u_count = sum(1 for e in unstable if get_fixed_flag(e) and get_fixed_flag(e)[1] == li)
    pts = sorted(fano_lines[li])
    pt_labels = [fano_labels[p] for p in pts]
    marker = ""
    if li == cartan_line:
        marker = " [CARTAN]"
    elif s_count > 0 and u_count == 0:
        marker = " STABLE"
    elif u_count > 0 and s_count == 0:
        marker = " UNSTABLE"
    elif s_count == 0 and u_count == 0:
        marker = " EMPTY"
    else:
        marker = " MIXED"
    print(f"  L{li} ({{{', '.join(pt_labels)}}}): stable={s_count}, unstable={u_count}{marker}")

# ============================================================
# STEP 6: Maximal abelian subalgebra
# ============================================================
print("\n" + "-" * 70)
print("  MAXIMAL ABELIAN SUBALGEBRA")
print("-" * 70)
from itertools import combinations

max_size = 0
max_sets = []
for size in range(len(line9), 0, -1):
    for subset in combinations(line9, size):
        if all(mul[a, mul[b, mul[a, b]]] == ID for a in subset for b in subset):
            if size > max_size:
                max_size = size
                max_sets = [subset]
            elif size == max_size:
                max_sets.append(subset)
    if max_sets:
        break

print(f"\n  Maximal commuting subsets (size {max_size}):")
for mc in max_sets:
    common_pts = None
    for e in mc:
        perm = fano_perm(e)
        fps = set(fano_fixed_points(perm))
        if common_pts is None:
            common_pts = fps
        else:
            common_pts &= fps
    common_label = [fano_labels[p] for p in sorted(common_pts)]
    is_cartan = (common_pts == set(fano_lines[cartan_line]))
    print(f"  {mc}: common fixed pts = {common_label}, Cartan? {is_cartan}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("  RESOLUTION SUMMARY")
print("=" * 70)
print(f"""
  Generator pair: (32, 1), gen_set = {psl_gen_set}
  Singletons: {len(singletons)} = {len(flag12)} flags + {len(line9)} lines

  COMMUTATORS (9 line involutions, {total} off-diagonal):
    ID:  {comm_landing.get('ID', 0)}
    W26: {comm_landing.get('W26', 0)}
    F36: {comm_landing.get('F36', 0)}
    B31: {comm_landing.get('B31', 0)}
    T75: {comm_landing.get('T75', 0)}
    W26:T75 = {w}:{t} = {w//gcd(w,t) if t else '?'}:{t//gcd(w,t) if t else '?'}
    3:8 MATCH: {w//gcd(w,t) == 3 and t//gcd(w,t) == 8 if t else False}

  T75 ORBITS HIT: {len(orbits_hit)}/14 (target 8 = dim SU(3))

  STABILITY CRITERION: {'HOLDS' if criterion else 'FAILS'}
    Stable ({len(stable)}): all OFF Cartan
    Unstable ({len(unstable)}): all ON Cartan
""")
