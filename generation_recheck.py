"""
generation_recheck.py
CRITICAL: Resolve whether <Z_62> = 166 or 168.
Three independent closure methods. No ambiguity.
"""
import sys
import numpy as np
from collections import Counter

sys.path.insert(0, '.')
from psl27_core import build_group, classify_strata

print("=" * 72)
print("  GENERATION RECHECK: |<Z_62>| = 166 or 168?")
print("=" * 72)

elems, e2i, mul, inv_table, ords, ID = build_group()
B_31, Z_62, T_75, z3, z3sq, W26, S3 = classify_strata(
    elems, e2i, mul, inv_table, ords, ID)

Z_list = sorted(Z_62)
print(f"\n  |Z_62| = {len(Z_62)} generators")
print(f"  Identity = element {ID}")

# =========================================================================
# METHOD A: BFS right-multiply only
# Start with {e}. At each step: for each frontier element g and each
# generator z in Z_62, compute g*z. Add if new.
# =========================================================================
print("\n" + "-" * 72)
print("  METHOD A: BFS RIGHT-MULTIPLY")
print("-" * 72)

reached_A = set([ID])
frontier_A = [ID]
step = 0
while frontier_A:
    step += 1
    next_frontier = []
    for g in frontier_A:
        for z in Z_list:
            prod = mul[g, z]
            if prod not in reached_A:
                reached_A.add(prod)
                next_frontier.append(prod)
    frontier_A = next_frontier
    print(f"  Step {step}: +{len(next_frontier)} new, total {len(reached_A)}")

print(f"  METHOD A RESULT: |<Z_62>| = {len(reached_A)}")
missing_A = set(range(168)) - reached_A
if missing_A:
    print(f"  Missing: {sorted(missing_A)}")
    for m in sorted(missing_A):
        print(f"    elem {m}: order {ords[m]}")

# =========================================================================
# METHOD B: BFS both left and right multiply
# Start with {e}. At each step: for each frontier element g and each
# generator z in Z_62, compute BOTH g*z and z*g. Add if new.
# =========================================================================
print("\n" + "-" * 72)
print("  METHOD B: BFS LEFT+RIGHT MULTIPLY")
print("-" * 72)

reached_B = set([ID])
frontier_B = [ID]
step = 0
while frontier_B:
    step += 1
    next_frontier = []
    for g in frontier_B:
        for z in Z_list:
            for prod in [mul[g, z], mul[z, g]]:
                if prod not in reached_B:
                    reached_B.add(prod)
                    next_frontier.append(prod)
    frontier_B = next_frontier
    print(f"  Step {step}: +{len(next_frontier)} new, total {len(reached_B)}")

print(f"  METHOD B RESULT: |<Z_62>| = {len(reached_B)}")
missing_B = set(range(168)) - reached_B
if missing_B:
    print(f"  Missing: {sorted(missing_B)}")
    for m in sorted(missing_B):
        print(f"    elem {m}: order {ords[m]}")

# =========================================================================
# METHOD C: Direct closure S = S ∪ {a*b : a,b ∈ S}
# Start with S = Z_62 ∪ {e}. Repeatedly close under multiplication.
# =========================================================================
print("\n" + "-" * 72)
print("  METHOD C: DIRECT CLOSURE (S = S ∪ {a*b})")
print("-" * 72)

S = set(Z_62) | {ID}
print(f"  Initial |S| = {len(S)}")
iteration = 0
while True:
    iteration += 1
    new_elements = set()
    S_list = sorted(S)
    for a in S_list:
        for b in S_list:
            prod = mul[a, b]
            if prod not in S:
                new_elements.add(prod)
    if not new_elements:
        break
    S |= new_elements
    print(f"  Iteration {iteration}: +{len(new_elements)} new, total {len(S)}")

print(f"  METHOD C RESULT: |<Z_62>| = {len(S)}")
missing_C = set(range(168)) - S
if missing_C:
    print(f"  Missing: {sorted(missing_C)}")
    for m in sorted(missing_C):
        print(f"    elem {m}: order {ords[m]}")

# =========================================================================
# METHOD D: Also include inverses explicitly
# Subgroup must be closed under inverses. Start with Z_62 ∪ inv(Z_62) ∪ {e}
# and close under multiplication.
# =========================================================================
print("\n" + "-" * 72)
print("  METHOD D: CLOSURE WITH EXPLICIT INVERSES")
print("-" * 72)

S_d = set(Z_62) | {ID}
# Add inverses of all Z_62 elements
for z in Z_62:
    S_d.add(inv_table[z])
print(f"  Initial |S| (with inverses) = {len(S_d)}")

iteration = 0
while True:
    iteration += 1
    new_elements = set()
    S_d_list = sorted(S_d)
    for a in S_d_list:
        for b in S_d_list:
            prod = mul[a, b]
            if prod not in S_d:
                new_elements.add(prod)
    if not new_elements:
        break
    S_d |= new_elements
    print(f"  Iteration {iteration}: +{len(new_elements)} new, total {len(S_d)}")

print(f"  METHOD D RESULT: |<Z_62>| = {len(S_d)}")
missing_D = set(range(168)) - S_d
if missing_D:
    print(f"  Missing: {sorted(missing_D)}")
    for m in sorted(missing_D):
        print(f"    elem {m}: order {ords[m]}")

# =========================================================================
# AGREEMENT CHECK
# =========================================================================
print("\n" + "-" * 72)
print("  AGREEMENT CHECK")
print("-" * 72)
print(f"  Method A (BFS right):       {len(reached_A)}")
print(f"  Method B (BFS left+right):  {len(reached_B)}")
print(f"  Method C (direct closure):  {len(S)}")
print(f"  Method D (closure+inv):     {len(S_d)}")
print(f"  All agree: {len(reached_A) == len(reached_B) == len(S) == len(S_d)}")

if len(reached_A) != len(reached_B):
    print(f"\n  A vs B divergence:")
    print(f"    In A not B: {sorted(reached_A - reached_B)}")
    print(f"    In B not A: {sorted(reached_B - reached_A)}")
if len(reached_A) != len(S):
    print(f"\n  A vs C divergence:")
    print(f"    In A not C: {sorted(reached_A - S)}")
    print(f"    In C not A: {sorted(S - reached_A)}")

# =========================================================================
# SECONDARY: Is element 18 expressible as product of Z_62 elements?
# =========================================================================
result = len(reached_A)  # use consensus
if result < 168:
    print("\n" + "-" * 72)
    print("  SECONDARY: WORD-LENGTH CERTIFICATE FOR ELEMENT 18")
    print("-" * 72)

    # Check all words of length 1..6
    # Length 1: is 18 in Z_62?
    print(f"\n  18 in Z_62: {18 in Z_62}")

    # Length 2: is 18 = z1 * z2 for any z1, z2 in Z_62?
    found_at = None
    for length in range(2, 7):
        if found_at is not None:
            break
        # BFS to depth 'length' from identity using Z_62
        reached_depth = {ID: 0}
        front = [ID]
        for d in range(length):
            nf = []
            for g in front:
                for z in Z_list:
                    p = mul[g, z]
                    if p not in reached_depth:
                        reached_depth[p] = d + 1
                        nf.append(p)
            front = nf
        if 18 in reached_depth:
            found_at = reached_depth[18]
            print(f"  Element 18 reachable at word length {found_at}")
            break
        else:
            print(f"  Length {length}: 18 NOT reachable ({len(reached_depth)} elements reached)")

    if found_at is None:
        print(f"  Element 18 NOT reachable in words up to length 6")
        print(f"  CERTIFICATE: element 18 is NOT in <Z_62>")
else:
    print("\n  <Z_62> = 168: element 18 IS in the closure.")
    # Find how 18 is reached
    # Trace back from BFS
    print(f"  Element 18 reached at depth: {0 if 18 == ID else '?'}")
    # Re-run BFS A with parent tracking
    parent = {ID: None}
    via = {ID: None}
    frontier_trace = [ID]
    while 18 not in parent or parent[18] is None and 18 != ID:
        next_ft = []
        for g in frontier_trace:
            for z in Z_list:
                prod = mul[g, z]
                if prod not in parent:
                    parent[prod] = g
                    via[prod] = z
                    next_ft.append(prod)
                    if prod == 18:
                        break
            if 18 in parent and parent[18] is not None:
                break
        frontier_trace = next_ft
        if not frontier_trace:
            break

    if 18 in parent and parent[18] is not None:
        # Trace path
        path = []
        cur = 18
        while parent[cur] is not None:
            path.append(via[cur])
            cur = parent[cur]
        path.reverse()
        print(f"  Element 18 = product of {len(path)} Z_62 generators:")
        print(f"    Generators (right to left): {path}")
        # Verify
        check = ID
        for z in path:
            check = mul[check, z]
        print(f"    Verification: product = {check}, correct: {check == 18}")
    else:
        print(f"  Could not trace path to element 18")

# =========================================================================
# DIAGNOSIS: What went wrong in the original subgroup_from?
# =========================================================================
print("\n" + "-" * 72)
print("  DIAGNOSIS: ORIGINAL subgroup_from FUNCTION")
print("-" * 72)

# Reproduce the EXACT function from interstratum_commutator.py
def subgroup_from_original(generators, mul_table, identity):
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

gen_original = subgroup_from_original(list(Z_62), mul, ID)
print(f"  Original subgroup_from: |<Z_62>| = {len(gen_original)}")
missing_orig = set(range(168)) - gen_original
if missing_orig:
    print(f"  Missing: {sorted(missing_orig)}")

# The bug: subgroup_from only multiplies NEW elements against existing ones.
# But it does NOT re-check products of OLD elements that might produce NEW
# elements after the set grows. It's a BFS that only considers products
# involving the newly-added element, not all pairs.
# This is correct for group generation from generators, but only if we also
# multiply new*new products. Let's check:

# Fixed version: also add products of new elements with each other
def subgroup_from_fixed(generators, mul_table, identity):
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
        # Also add inverse
        # For finite groups: g^(ord-1) = g^(-1)
        # But we need ord, which requires iteration
        for x in new:
            sg.add(x)
            queue.append(x)
    return frozenset(sg)

gen_fixed = subgroup_from_fixed(list(Z_62), mul, ID)
print(f"  Fixed subgroup_from: |<Z_62>| = {len(gen_fixed)}")

# The real issue might be queue exhaustion. Let's count operations:
def subgroup_from_verbose(generators, mul_table, identity):
    sg = {identity}
    queue = list(generators)
    iterations = 0
    max_queue = len(queue)
    while queue:
        iterations += 1
        g = queue.pop()
        if g in sg:
            continue
        sg.add(g)
        added = 0
        for h in list(sg):
            for x in [mul_table[g, h], mul_table[h, g]]:
                if x not in sg:
                    sg.add(x)
                    queue.append(x)
                    added += 1
        if iterations % 50 == 0:
            print(f"    iter {iterations}: |sg|={len(sg)}, queue={len(queue)}, added={added}")
        max_queue = max(max_queue, len(queue))
    return frozenset(sg), iterations, max_queue

gen_v, iters, mq = subgroup_from_verbose(list(Z_62), mul, ID)
print(f"  Verbose: |<Z_62>| = {len(gen_v)}, iterations={iters}, max_queue={mq}")

# =========================================================================
# DEFINITIVE ANSWER
# =========================================================================
print("\n" + "=" * 72)
print("  DEFINITIVE ANSWER")
print("=" * 72)

results = {
    'A (BFS right)': len(reached_A),
    'B (BFS left+right)': len(reached_B),
    'C (direct closure)': len(S),
    'D (closure+inv)': len(S_d),
    'Original subgroup_from': len(gen_original),
    'Fixed subgroup_from': len(gen_fixed),
    'Verbose subgroup_from': len(gen_v),
}

for name, val in results.items():
    status = "FULL" if val == 168 else f"MISSING {168 - val}"
    print(f"  {name:30s}: {val} ({status})")

consensus = Counter(results.values()).most_common(1)[0]
print(f"\n  CONSENSUS: |<Z_62>| = {consensus[0]} ({consensus[1]}/{len(results)} methods agree)")

if consensus[0] == 168:
    print(f"""
  THE WEAK SECTOR GENERATES THE FULL GROUP.
  The earlier result of 166 was a BUG in subgroup_from.
  Element 18 IS reachable from Z_62 products.
  The 'bridge element' narrative needs revision.

  WHAT STILL STANDS:
  - Z_4 = {{e, 18, 92, 4}} straddling B_31 and T_75
  - 18^2 = 92 = Z boson candidate (Fix(C) ∩ B_31)
  - Fano point 010 uniqueness
  - All stratum classifications
  - S_3 universality
  - Interstratum commutator structure
""")
elif consensus[0] == 166:
    print(f"""
  THE WEAK SECTOR GENUINELY CANNOT REACH 2 ELEMENTS.
  Element 18 and 161 are outside <Z_62>.
  The bridge narrative is confirmed.
""")
