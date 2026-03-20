"""
psl27_core.py — Shared PSL(2,7) = GL(3, F_2) infrastructure for Paper 13.

Builds all 168 elements, multiplication table, inverses, orders,
conjugacy classes, and the three-stratum decomposition:
  B_31  (binary, 31 elements)
  Z_62  (Z_3-boundary, 62 elements)
  T_75  (purely ternary, 75 elements)
"""
import numpy as np
from itertools import product as iproduct
from collections import Counter

# ── F_2 linear algebra ──────────────────────────────────────────────────────

def det_f2(A):
    n = A.shape[0]
    M = A.copy() % 2
    for col in range(n):
        pivot = next((r for r in range(col, n) if M[r, col] == 1), None)
        if pivot is None:
            return 0
        M[[col, pivot]] = M[[pivot, col]]
        for r in range(col + 1, n):
            if M[r, col] == 1:
                M[r] = (M[r] + M[col]) % 2
    return 1


def mat_inv_f2(A):
    n = A.shape[0]
    aug = np.hstack([A.copy(), np.eye(n, dtype=int)]) % 2
    for col in range(n):
        pivot = next((r for r in range(col, n) if aug[r, col] == 1), None)
        if pivot is None:
            return None
        aug[[col, pivot]] = aug[[pivot, col]]
        for r in range(n):
            if r != col and aug[r, col] == 1:
                aug[r] = (aug[r] + aug[col]) % 2
    return aug[:, n:] % 2


def mat_key(M):
    return tuple(M.flatten().tolist())


# ── Build PSL(2,7) = GL(3, F_2) ─────────────────────────────────────────────

def build_group():
    """Return (elems, e2i, mul, inv_table, ords, ID)."""
    elems = [np.array(e, dtype=int).reshape(3, 3)
             for e in iproduct([0, 1], repeat=9)
             if det_f2(np.array(e).reshape(3, 3))]
    assert len(elems) == 168, f"Expected 168 elements, got {len(elems)}"

    e2i = {mat_key(M): i for i, M in enumerate(elems)}
    ID = e2i[mat_key(np.eye(3, dtype=int))]

    # Multiplication table
    mul = np.zeros((168, 168), dtype=int)
    for i, A in enumerate(elems):
        for j, B in enumerate(elems):
            mul[i, j] = e2i[mat_key((A @ B) % 2)]

    # Inverse table
    inv_table = np.zeros(168, dtype=int)
    for i in range(168):
        inv_table[i] = e2i[mat_key(mat_inv_f2(elems[i]))]

    # Orders
    ords = np.zeros(168, dtype=int)
    for i in range(168):
        cur = ID
        for k in range(1, 169):
            cur = mul[cur, i]
            if cur == ID:
                ords[i] = k
                break

    return elems, e2i, mul, inv_table, ords, ID


def classify_strata(elems, e2i, mul, inv_table, ords, ID):
    """
    Return (B_31, Z_62, T_75, z3, z3sq, weinberg_26, stab_s3).

    B_31: binary stratum (31 elements, orders 1/2/4 with Z3-arch property)
    Z_62: Z3-boundary (62 elements)
    T_75: purely ternary (75 elements)
    """
    # Pick a Z3 generator
    z3 = next(i for i, o in enumerate(ords) if o == 3)
    z3sq = mul[z3, z3]

    # Binary stratum: orders that are powers of 2
    binary_stratum = {i for i in range(168) if ords[i] in {1, 2, 4}}

    # arch_31: binary elements whose Z3-orbit has exactly 1 element in binary
    B_31 = {b for b in binary_stratum
            if sum(1 for x in [b, mul[z3, b], mul[z3sq, b]]
                   if x in binary_stratum) == 1}
    assert len(B_31) == 31, f"|B| = {len(B_31)}, expected 31"

    # Z3-closure of B_31
    z3_closure = set()
    for b in B_31:
        z3_closure.update([b, mul[z3, b], mul[z3sq, b]])
    Z_62 = z3_closure - B_31
    assert len(Z_62) == 62, f"|Z| = {len(Z_62)}, expected 62"

    # Purely ternary
    T_75 = set(range(168)) - B_31 - Z_62
    assert len(T_75) == 75, f"|T| = {len(T_75)}, expected 75"

    # Weinberg 26: order-3 elements in Z3-boundary
    weinberg_26 = {i for i in Z_62 if ords[i] == 3}
    assert len(weinberg_26) == 26, f"|W| = {len(weinberg_26)}, expected 26"

    # S3 stabiliser of T_75
    stab_s3 = [g for g in range(168)
               if all(mul[g, mul[t, inv_table[g]]] in T_75 for t in T_75)]
    assert len(stab_s3) == 6, f"|S3| = {len(stab_s3)}, expected 6"

    return B_31, Z_62, T_75, z3, z3sq, weinberg_26, stab_s3


def conjugacy_classes(mul, inv_table):
    """Return list of conjugacy classes (each a sorted list of element indices)."""
    assigned = [False] * 168
    classes = []
    for i in range(168):
        if assigned[i]:
            continue
        cls = set()
        for g in range(168):
            cls.add(mul[g, mul[i, inv_table[g]]])
        for j in cls:
            assigned[j] = True
        classes.append(sorted(cls))
    classes.sort(key=lambda c: (len(c), c[0]))
    return classes


def print_order_distribution(ords, label="Group"):
    dist = sorted(Counter(int(ords[i]) for i in range(168)).items())
    print(f"  {label} order distribution: {dict(dist)}")


if __name__ == "__main__":
    print("Building PSL(2,7) = GL(3, F_2)...")
    elems, e2i, mul, inv_table, ords, ID = build_group()
    B_31, Z_62, T_75, z3, z3sq, W26, S3 = classify_strata(
        elems, e2i, mul, inv_table, ords, ID)
    print(f"  |G| = 168, |B| = {len(B_31)}, |Z| = {len(Z_62)}, |T| = {len(T_75)}")
    print_order_distribution(ords)
    print(f"  |Weinberg| = {len(W26)}, |S3| = {len(S3)}")
    print("  All assertions passed.")
