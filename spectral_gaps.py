"""
spectral_gaps.py
Compute spectral gaps of the Cayley graph restricted to each stratum.
Test against h*(E6):h*(B6):h*(C6) = 12:11:7.

The physical conjecture: the force hierarchy is set by spectral gap ratios
of the three strata under the S_3 meta-symmetry.
"""
import sys
import numpy as np
from collections import Counter, deque
from scipy import sparse
from scipy.sparse.linalg import eigsh

sys.path.insert(0, '.')
from psl27_core import build_group, classify_strata

print("=" * 72)
print("  SPECTRAL GAPS: STRATUM CAYLEY SUBGRAPHS")
print("=" * 72)

elems, e2i, mul, inv_table, ords, ID = build_group()
B_31, Z_62, T_75, z3, z3sq, W26, S3 = classify_strata(
    elems, e2i, mul, inv_table, ords, ID)

# =========================================================================
# GENERATING SETS
# =========================================================================

# 1. S_3 generating set (the meta-symmetry)
s3_gen = sorted(S3)
print(f"\n  S_3 stabiliser: {s3_gen} (orders {[int(ords[g]) for g in s3_gen]})")

# 2. Full PSL(2,7) generating set (minimal)
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

psl_gen_pair = find_generators(mul, ords, ID)
psl_gen = sorted(set(psl_gen_pair + [inv_table[g] for g in psl_gen_pair]))
print(f"  PSL(2,7) generators: {psl_gen_pair} (orders {[int(ords[g]) for g in psl_gen_pair]})")

# =========================================================================
# HELPER: Build adjacency and Laplacian for a stratum
# =========================================================================

def build_stratum_graph(stratum, generators, mul_table, inv_table_arr):
    """
    Build the induced subgraph on a stratum where edges are defined by
    conjugation: x ~ g*x*g^{-1} for g in generators.

    Returns adjacency matrix and list of stratum elements.
    """
    elements = sorted(stratum)
    n = len(elements)
    elem_to_idx = {e: i for i, e in enumerate(elements)}

    adj = np.zeros((n, n), dtype=float)

    for e in elements:
        i = elem_to_idx[e]
        for g in generators:
            # Conjugation: g * e * g^{-1}
            image = mul_table[g, mul_table[e, inv_table_arr[g]]]
            if image in elem_to_idx:
                j = elem_to_idx[image]
                if i != j:
                    adj[i, j] = 1
                    adj[j, i] = 1

    return adj, elements


def build_stratum_cayley(stratum, generators, mul_table, inv_table_arr):
    """
    Build the induced Cayley subgraph on a stratum where edges connect
    elements related by LEFT or RIGHT multiplication by generators,
    IF both endpoints are in the stratum.
    """
    elements = sorted(stratum)
    n = len(elements)
    elem_to_idx = {e: i for i, e in enumerate(elements)}

    adj = np.zeros((n, n), dtype=float)

    for e in elements:
        i = elem_to_idx[e]
        for g in generators:
            # Right multiplication
            r = mul_table[e, g]
            if r in elem_to_idx:
                j = elem_to_idx[r]
                if i != j:
                    adj[i, j] = 1
                    adj[j, i] = 1
            # Left multiplication
            l = mul_table[g, e]
            if l in elem_to_idx:
                j = elem_to_idx[l]
                if i != j:
                    adj[i, j] = 1
                    adj[j, i] = 1

    return adj, elements


def spectral_gap(adj):
    """Compute spectral gap of the graph Laplacian."""
    n = adj.shape[0]
    degree = adj.sum(axis=1)
    laplacian = np.diag(degree) - adj

    eigenvalues = np.linalg.eigvalsh(laplacian)
    eigenvalues = np.sort(eigenvalues)

    # lambda_0 should be ~0 (connected component)
    lambda_0 = eigenvalues[0]
    lambda_1 = eigenvalues[1] if n > 1 else 0

    return eigenvalues, lambda_0, lambda_1


def print_graph_stats(adj, elements, label):
    """Print statistics about the graph."""
    n = adj.shape[0]
    n_edges = int(adj.sum() / 2)
    degrees = adj.sum(axis=1)
    print(f"\n  {label}:")
    print(f"    Nodes: {n}, Edges: {n_edges}")
    print(f"    Degree: min={int(degrees.min())}, max={int(degrees.max())}, "
          f"mean={degrees.mean():.2f}")

    # Check connectivity
    visited = {0}
    queue = deque([0])
    while queue:
        node = queue.popleft()
        for nbr in range(n):
            if adj[node, nbr] > 0 and nbr not in visited:
                visited.add(nbr)
                queue.append(nbr)
    connected = len(visited) == n
    print(f"    Connected: {connected} ({len(visited)}/{n} reachable)")

    return connected


# =========================================================================
# 1. CONJUGATION GRAPH (S_3 ACTION)
# =========================================================================
print("\n" + "-" * 72)
print("  1. SPECTRAL GAPS: S_3 CONJUGATION GRAPH")
print("-" * 72)

strata = [('B_31', B_31), ('Z_62', Z_62), ('T_75', T_75)]
results_conj = {}

for label, stratum in strata:
    adj, elts = build_stratum_graph(stratum, s3_gen, mul, inv_table)
    connected = print_graph_stats(adj, elts, f"{label} (S_3 conjugation)")

    eigenvalues, l0, l1 = spectral_gap(adj)
    print(f"    lambda_0 = {l0:.6f}")
    print(f"    lambda_1 = {l1:.6f} (spectral gap)")
    print(f"    lambda_max = {eigenvalues[-1]:.6f}")

    # Number of zero eigenvalues = number of connected components
    n_zero = sum(1 for ev in eigenvalues if abs(ev) < 1e-10)
    print(f"    Zero eigenvalues (components): {n_zero}")

    results_conj[label] = {
        'lambda_1': l1, 'eigenvalues': eigenvalues,
        'connected': connected, 'n_components': n_zero
    }

# Ratio test
print(f"\n  Spectral gap ratios (S_3 conjugation):")
gaps = {k: v['lambda_1'] for k, v in results_conj.items()}
if all(g > 1e-10 for g in gaps.values()):
    vals = np.array([gaps['B_31'], gaps['Z_62'], gaps['T_75']])
    normalised = vals * 30.0 / vals.sum()
    print(f"    B: {gaps['B_31']:.6f}, Z: {gaps['Z_62']:.6f}, T: {gaps['T_75']:.6f}")
    print(f"    Normalised (sum=30): B={normalised[0]:.3f}, Z={normalised[1]:.3f}, T={normalised[2]:.3f}")
    print(f"    Target (12:11:7):    12.000, 11.000, 7.000")
    print(f"    Residual: {np.linalg.norm(normalised - np.array([12,11,7])):.4f}")

    # All 6 permutations
    from itertools import permutations
    best_resid = float('inf')
    best_perm = None
    target = np.array([12, 11, 7])
    for perm in permutations([0, 1, 2]):
        perm_vals = np.array([vals[perm[0]], vals[perm[1]], vals[perm[2]]])
        perm_norm = perm_vals * 30.0 / perm_vals.sum()
        resid = np.linalg.norm(perm_norm - target)
        if resid < best_resid:
            best_resid = resid
            best_perm = perm
    labels_list = ['B', 'Z', 'T']
    algebras = ['E6(12)', 'B6(11)', 'C6(7)']
    print(f"    Best permutation (residual {best_resid:.4f}):")
    for j, alg in enumerate(algebras):
        print(f"      {alg} <-> {labels_list[best_perm[j]]} (gap={vals[best_perm[j]]:.6f})")
else:
    print(f"    WARNING: Some spectral gaps are zero (disconnected graphs)")
    for k, v in gaps.items():
        print(f"      {k}: gap = {v:.6f}")

# =========================================================================
# 2. CAYLEY GRAPH (PSL(2,7) GENERATORS, MULTIPLICATION)
# =========================================================================
print("\n" + "-" * 72)
print("  2. SPECTRAL GAPS: PSL(2,7) CAYLEY SUBGRAPH (MULTIPLICATION)")
print("-" * 72)

results_cayley = {}

for label, stratum in strata:
    adj, elts = build_stratum_cayley(stratum, psl_gen, mul, inv_table)
    connected = print_graph_stats(adj, elts, f"{label} (PSL generators, mult)")

    eigenvalues, l0, l1 = spectral_gap(adj)
    print(f"    lambda_0 = {l0:.6f}")
    print(f"    lambda_1 = {l1:.6f} (spectral gap)")
    print(f"    lambda_max = {eigenvalues[-1]:.6f}")

    n_zero = sum(1 for ev in eigenvalues if abs(ev) < 1e-10)
    print(f"    Zero eigenvalues (components): {n_zero}")

    results_cayley[label] = {
        'lambda_1': l1, 'eigenvalues': eigenvalues,
        'connected': connected, 'n_components': n_zero
    }

# Ratio test
print(f"\n  Spectral gap ratios (PSL Cayley):")
gaps_c = {k: v['lambda_1'] for k, v in results_cayley.items()}
if all(g > 1e-10 for g in gaps_c.values()):
    vals = np.array([gaps_c['B_31'], gaps_c['Z_62'], gaps_c['T_75']])
    normalised = vals * 30.0 / vals.sum()
    print(f"    B: {gaps_c['B_31']:.6f}, Z: {gaps_c['Z_62']:.6f}, T: {gaps_c['T_75']:.6f}")
    print(f"    Normalised (sum=30): B={normalised[0]:.3f}, Z={normalised[1]:.3f}, T={normalised[2]:.3f}")
    print(f"    Target (12:11:7):    12.000, 11.000, 7.000")
    resid = np.linalg.norm(normalised - np.array([12,11,7]))
    print(f"    Residual: {resid:.4f}")

    best_resid = float('inf')
    best_perm = None
    target = np.array([12, 11, 7])
    for perm in permutations([0, 1, 2]):
        perm_vals = np.array([vals[perm[0]], vals[perm[1]], vals[perm[2]]])
        perm_norm = perm_vals * 30.0 / perm_vals.sum()
        r = np.linalg.norm(perm_norm - target)
        if r < best_resid:
            best_resid = r
            best_perm = perm
    labels_list = ['B', 'Z', 'T']
    algebras = ['E6(12)', 'B6(11)', 'C6(7)']
    print(f"    Best permutation (residual {best_resid:.4f}):")
    for j, alg in enumerate(algebras):
        print(f"      {alg} <-> {labels_list[best_perm[j]]} (gap={vals[best_perm[j]]:.6f})")
else:
    print(f"    WARNING: Some spectral gaps are zero")
    for k, v in gaps_c.items():
        print(f"      {k}: gap = {v:.6f}")

# =========================================================================
# 3. CAYLEY GRAPH WITH S_3 GENERATORS (MULTIPLICATION)
# =========================================================================
print("\n" + "-" * 72)
print("  3. SPECTRAL GAPS: S_3 CAYLEY SUBGRAPH (MULTIPLICATION)")
print("-" * 72)

results_s3_cayley = {}

for label, stratum in strata:
    adj, elts = build_stratum_cayley(stratum, s3_gen, mul, inv_table)
    connected = print_graph_stats(adj, elts, f"{label} (S_3 generators, mult)")

    eigenvalues, l0, l1 = spectral_gap(adj)
    print(f"    lambda_0 = {l0:.6f}")
    print(f"    lambda_1 = {l1:.6f} (spectral gap)")

    n_zero = sum(1 for ev in eigenvalues if abs(ev) < 1e-10)
    print(f"    Zero eigenvalues (components): {n_zero}")

    results_s3_cayley[label] = {
        'lambda_1': l1, 'eigenvalues': eigenvalues,
        'connected': connected, 'n_components': n_zero
    }

# Ratio test
print(f"\n  Spectral gap ratios (S_3 Cayley mult):")
gaps_s3c = {k: v['lambda_1'] for k, v in results_s3_cayley.items()}
if all(g > 1e-10 for g in gaps_s3c.values()):
    vals = np.array([gaps_s3c['B_31'], gaps_s3c['Z_62'], gaps_s3c['T_75']])
    normalised = vals * 30.0 / vals.sum()
    print(f"    B: {gaps_s3c['B_31']:.6f}, Z: {gaps_s3c['Z_62']:.6f}, T: {gaps_s3c['T_75']:.6f}")
    print(f"    Normalised (sum=30): B={normalised[0]:.3f}, Z={normalised[1]:.3f}, T={normalised[2]:.3f}")
    print(f"    Target (12:11:7):    12.000, 11.000, 7.000")
    resid = np.linalg.norm(normalised - np.array([12,11,7]))
    print(f"    Residual: {resid:.4f}")

    best_resid = float('inf')
    best_perm = None
    for perm in permutations([0, 1, 2]):
        perm_vals = np.array([vals[perm[0]], vals[perm[1]], vals[perm[2]]])
        perm_norm = perm_vals * 30.0 / perm_vals.sum()
        r = np.linalg.norm(perm_norm - target)
        if r < best_resid:
            best_resid = r
            best_perm = perm
    labels_list = ['B', 'Z', 'T']
    algebras = ['E6(12)', 'B6(11)', 'C6(7)']
    print(f"    Best permutation (residual {best_resid:.4f}):")
    for j, alg in enumerate(algebras):
        print(f"      {alg} <-> {labels_list[best_perm[j]]} (gap={vals[best_perm[j]]:.6f})")
else:
    print(f"    WARNING: Some spectral gaps are zero")
    for k, v in gaps_s3c.items():
        print(f"      {k}: gap = {v:.6f}")

# =========================================================================
# 4. BIPARTITE COUPLING GRAPHS
# =========================================================================
print("\n" + "-" * 72)
print("  4. BIPARTITE COUPLING GAPS: B_31<->Z_62 AND Z_62<->T_75")
print("-" * 72)

def build_bipartite_graph(stratum_a, stratum_b, generators, mul_table, inv_table_arr):
    """
    Build bipartite graph between two strata.
    Edges: a in stratum_a connected to b in stratum_b if
    g*a*g^{-1} = b or a*g = b or g*a = b for some generator g.
    """
    elems_a = sorted(stratum_a)
    elems_b = sorted(stratum_b)
    na, nb = len(elems_a), len(elems_b)
    idx_a = {e: i for i, e in enumerate(elems_a)}
    idx_b = {e: i for i, e in enumerate(elems_b)}

    # Biadjacency matrix B: B[i,j] = 1 if a_i connected to b_j
    biadj = np.zeros((na, nb), dtype=float)

    for a in elems_a:
        i = idx_a[a]
        for g in generators:
            # Left mult
            l = mul_table[g, a]
            if l in idx_b:
                biadj[i, idx_b[l]] = 1
            # Right mult
            r = mul_table[a, g]
            if r in idx_b:
                biadj[i, idx_b[r]] = 1
            # Conjugation
            c = mul_table[g, mul_table[a, inv_table_arr[g]]]
            if c in idx_b:
                biadj[i, idx_b[c]] = 1

    return biadj, elems_a, elems_b


for gen_label, generators in [("PSL generators", psl_gen), ("S_3 generators", s3_gen)]:
    print(f"\n  Using {gen_label}:")

    # B_31 <-> Z_62
    biadj_bz, _, _ = build_bipartite_graph(B_31, Z_62, generators, mul, inv_table)
    # Full bipartite Laplacian: [[D_a, -B], [-B^T, D_b]]
    na, nb = biadj_bz.shape
    D_a = np.diag(biadj_bz.sum(axis=1))
    D_b = np.diag(biadj_bz.sum(axis=0))
    L_bz = np.block([[D_a, -biadj_bz], [-biadj_bz.T, D_b]])
    eigs_bz = np.linalg.eigvalsh(L_bz)
    eigs_bz = np.sort(eigs_bz)
    n_zero_bz = sum(1 for ev in eigs_bz if abs(ev) < 1e-10)
    gap_bz = eigs_bz[n_zero_bz] if n_zero_bz < len(eigs_bz) else 0

    print(f"\n    B_31 <-> Z_62 bipartite:")
    print(f"      Edges: {int(biadj_bz.sum())}")
    print(f"      Zero eigenvalues: {n_zero_bz}")
    print(f"      Spectral gap: {gap_bz:.6f}")

    # Z_62 <-> T_75
    biadj_zt, _, _ = build_bipartite_graph(Z_62, T_75, generators, mul, inv_table)
    na, nb = biadj_zt.shape
    D_a = np.diag(biadj_zt.sum(axis=1))
    D_b = np.diag(biadj_zt.sum(axis=0))
    L_zt = np.block([[D_a, -biadj_zt], [-biadj_zt.T, D_b]])
    eigs_zt = np.linalg.eigvalsh(L_zt)
    eigs_zt = np.sort(eigs_zt)
    n_zero_zt = sum(1 for ev in eigs_zt if abs(ev) < 1e-10)
    gap_zt = eigs_zt[n_zero_zt] if n_zero_zt < len(eigs_zt) else 0

    print(f"\n    Z_62 <-> T_75 bipartite:")
    print(f"      Edges: {int(biadj_zt.sum())}")
    print(f"      Zero eigenvalues: {n_zero_zt}")
    print(f"      Spectral gap: {gap_zt:.6f}")

    if gap_bz > 1e-10 and gap_zt > 1e-10:
        print(f"\n    Bipartite gap ratio: B-Z / Z-T = {gap_bz/gap_zt:.4f}")
        print(f"    Physical prediction: W boson / Z boson mass ratio?")

# =========================================================================
# 5. FULL GROUP CAYLEY GRAPH WITH STRATUM-RESTRICTED LAPLACIAN
# =========================================================================
print("\n" + "-" * 72)
print("  5. FULL GROUP CAYLEY GRAPH: STRATUM-RESTRICTED SPECTRA")
print("-" * 72)

# Build full Cayley graph of PSL(2,7)
adj_full = np.zeros((168, 168), dtype=float)
for e in range(168):
    for g in psl_gen:
        r = mul[e, g]
        if r != e:
            adj_full[e, r] = 1
            adj_full[r, e] = 1
        l = mul[g, e]
        if l != e:
            adj_full[e, l] = 1
            adj_full[l, e] = 1

degree_full = adj_full.sum(axis=1)
L_full = np.diag(degree_full) - adj_full
eigs_full = np.linalg.eigvalsh(L_full)
eigs_full = np.sort(eigs_full)

print(f"\n  Full Cayley graph:")
print(f"    Nodes: 168, Edges: {int(adj_full.sum()/2)}")
print(f"    Degree range: {int(degree_full.min())}-{int(degree_full.max())}")
print(f"    lambda_0 = {eigs_full[0]:.6f}")
print(f"    lambda_1 = {eigs_full[1]:.6f} (spectral gap)")
print(f"    lambda_max = {eigs_full[-1]:.6f}")

# Project Laplacian onto each stratum (restriction = principal submatrix)
for label, stratum in strata:
    indices = sorted(stratum)
    L_sub = L_full[np.ix_(indices, indices)]
    eigs_sub = np.linalg.eigvalsh(L_sub)
    eigs_sub = np.sort(eigs_sub)

    # The smallest eigenvalue of the restricted Laplacian is NOT zero in general
    # (it measures how connected the stratum is within the full graph)
    print(f"\n  {label} restricted Laplacian:")
    print(f"    lambda_min = {eigs_sub[0]:.6f}")
    print(f"    lambda_1 = {eigs_sub[1]:.6f}")
    print(f"    lambda_max = {eigs_sub[-1]:.6f}")
    print(f"    First 5 eigenvalues: {eigs_sub[:5].round(4)}")

# =========================================================================
# SUMMARY
# =========================================================================
print("\n" + "=" * 72)
print("  SUMMARY: SPECTRAL GAP ANALYSIS")
print("=" * 72)

print(f"""
  Three graph constructions tested:

  1. S_3 CONJUGATION GRAPH (edges = S_3-conjugation within stratum):
     B_31: gap = {results_conj['B_31']['lambda_1']:.6f}, components = {results_conj['B_31']['n_components']}
     Z_62: gap = {results_conj['Z_62']['lambda_1']:.6f}, components = {results_conj['Z_62']['n_components']}
     T_75: gap = {results_conj['T_75']['lambda_1']:.6f}, components = {results_conj['T_75']['n_components']}

  2. PSL(2,7) CAYLEY SUBGRAPH (edges = mult by PSL generators, within stratum):
     B_31: gap = {results_cayley['B_31']['lambda_1']:.6f}, components = {results_cayley['B_31']['n_components']}
     Z_62: gap = {results_cayley['Z_62']['lambda_1']:.6f}, components = {results_cayley['Z_62']['n_components']}
     T_75: gap = {results_cayley['T_75']['lambda_1']:.6f}, components = {results_cayley['T_75']['n_components']}

  3. S_3 CAYLEY SUBGRAPH (edges = mult by S_3 elements, within stratum):
     B_31: gap = {results_s3_cayley['B_31']['lambda_1']:.6f}, components = {results_s3_cayley['B_31']['n_components']}
     Z_62: gap = {results_s3_cayley['Z_62']['lambda_1']:.6f}, components = {results_s3_cayley['Z_62']['n_components']}
     T_75: gap = {results_s3_cayley['T_75']['lambda_1']:.6f}, components = {results_s3_cayley['T_75']['n_components']}

  h* PREDICTION: 12 : 11 : 7 (E6 : B6 : C6)

  THE QUESTION: Does the spectral gap ratio match the force hierarchy?
""")
