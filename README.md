# The Standard Model Gauge Group from PSL(2,7)

Computational scripts for Paper 13 of the Merkabit series.

**Paper:** *The Standard Model Gauge Group from PSL(2,7): SU(3)x SU(2)xU(1) as Representation Theory of the Three-Stratum Decomposition of GL(3,F_2)*

**Author:** Selina Stenberg with Claude Anthropic (March 2026)

## Core Result

The Standard Model gauge group SU(3)xSU(2)xU(1) is derived from the representation theory of PSL(2,7) = GL(3,F_2), order 168, without imposing the gauge group as an input.

```
PSL(2,7) = B_31 + Z_62 + T_75   (168 = 31 + 62 + 75)

SU(3): tensor products 3x3-bar = 1+8, wedge^3(3) = 1 (exact)
       3 and 3-bar are the UNIQUE complex-Schur irreps (confinement)
SU(2): Z_62 anti-incidence (dim-7); W+/- as Z_2-bundle over B_13
U(1):  Z_3 grading of Z[omega]; photon fixes Fano point 010
```

## Three Exact Theorems

| # | Theorem | Statement |
|---|---------|-----------|
| 1 | S_3 Universality | S_3 preserves all strata with distinct characters: 8, 11, 14 orbits |
| 2 | Z_3-Boundary Sufficiency | Z_62 generates full PSL(2,7); weak sector algebraically complete |
| 3 | Confinement by Schur Indicator | 3 and 3-bar are the unique complex irreps; isolated colour unobservable |

## Gauge Boson Identification

| Boson | Location | Property |
|-------|----------|----------|
| W+/- | Z_62 anti-incidence | 13+13 Weinberg split, algebraic inverses under C |
| Z^0 | Fix(C) &cap; B_31 | Element 92 = 18^2, unique Z_4 midpoint |
| g (x8) | dim-8 adjoint on B_31 | Couples T_75 to B_31; real Schur |
| gamma | Z_3 grading of Z[omega] | Fixes Fano point 010; outside PSL(2,7); massless |

## Numerical Predictions

| Quantity | Formula | Prediction | Measured | Match |
|----------|---------|------------|----------|-------|
| sin^2(theta_W) | 3/13 | 0.23077 | 0.23122 | 0.19% |
| m_W | 47v/144 | 80.36 GeV | 80.377 GeV | 0.017% |
| m_Z | 47v sqrt(13)/(144 sqrt(10)) | 91.63 GeV | 91.19 GeV | 0.48% |
| alpha^-1 | Three-rung E_6/B_6/C_6 | 137.035999084 | 137.035999166 | 0.007 ppb |

## Scripts

### Core

| Script | Description |
|--------|-------------|
| `psl27_core.py` | Shared library: builds PSL(2,7), classifies strata B_31/Z_62/T_75, multiplication and inverse tables |
| `s3_universality.py` | S_3 preserves all strata; orbit counts 8/11/14 |
| `generation_recheck.py` | Four methods confirm Z_62 generates full PSL(2,7) |
| `z62_missing_elements.py` | BFS depth analysis of Z_62 closure |
| `confinement_theorem.py` | Schur indicators, tensor products, dim-8 restricted to S_3 |
| `psl27_representation_theory.py` | Character table, stratum decomposition into irreps |

### Gauge Structure

| Script | Description |
|--------|-------------|
| `order4_bridge.py` | Element 18 profile: Fano action, Z_4 chain, Z boson candidate |
| `z4_structure_proof.py` | Full Z_4 = {e, 18, 92, 4} characterisation; 010 isolation |
| `dim7_identification.py` | Dim-7 anti-incidence representation; SU(2) connection |
| `b31_orbit_structure.py` | 8 S_3-orbits of B_31; Fix(C) structure |
| `interstratum_commutator.py` | [T,Z] commutators; stratum leakage; simplicity confirmation |
| `automorphism_t75.py` | Aut(T_75) = S_3, not SU(3); SU(3) requires representation theory |

### Extended Analysis

| Script | Description |
|--------|-------------|
| `gauge_boson_ratio.py` | Commutator gauge ratio 3:8 (weak:strong) |
| `fano_core_36.py` | Fano plane core geometry; 36 neutral elements |
| `fano_flags_matter.py` | Flag/anti-flag structure; matter as incidence geometry |
| `lie_algebra_matter.py` | Lie algebra structure of matter sector |
| `matter_root_structure.py` | Root system analysis of B_31 |
| `neutral_current.py` | Neutral current sector analysis |
| `spectral_gaps.py` | Spectral gap eigenvalue analysis per stratum |
| `g2_orbit_correspondence.py` | G_2 orbit correspondence |
| `threshold_ratios.py` | Threshold ratio computations |
| `stratum_threshold_distances.py` | Inter-stratum distance metrics |
| `internal_orbit_thresholds.py` | Internal orbit threshold analysis |
| `resolve_inconsistency.py` | Resolution of Z_62 generation discrepancy |

## Dependencies

- Python 3.8+
- NumPy

## Companion Papers

- **Base document:** [The Merkabit](https://doi.org/10.5281/zenodo.18925475) (v4, March 2026)
- **Paper 1:** [alpha = 4/3 in Driven Coherent Systems](https://doi.org/10.5281/zenodo.18980026)
- **Paper 2:** [A Single Geometric Constant](https://doi.org/10.5281/zenodo.18981288)
- **Paper 5:** [Fusion Ignition from E_6 Geometry](https://doi.org/10.5281/zenodo.18984593)
- **Paper 6:** [Geometric Operator on the Eisenstein Lattice](https://doi.org/10.5281/zenodo.19075162)
- **Paper 7:** [The Riemann Zeros as Collapse Events](https://doi.org/10.5281/zenodo.19053965)
- **Paper 8:** [The Merkabit Architecture and the Klein Quartic](https://doi.org/10.5281/zenodo.19066587)
- **Paper 9:** [The Yang-Mills Mass Gap](https://doi.org/10.5281/zenodo.19144885)
- **Paper 10:** [One Group, Four Faces](https://doi.org/10.5281/zenodo.19147064)
- **Paper 11:** [The Standard Model as S_3-Invariant Decomposition](https://doi.org/10.5281/zenodo.19150963)
- **Paper 12:** [The W Boson as Algebraic Inversion](https://doi.org/10.5281/zenodo.19153206)

## License

MIT License. See [LICENSE](LICENSE).
