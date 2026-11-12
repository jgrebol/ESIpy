# Glynn-based MCI in ESIpy: methods, validation and results

Branch dev-mciaprox. Code in `esipy/mci.py`: `compute_mci` and `mci_approx` use Glynn's identity (same signatures);
the DFS is kept as legacy code (`compute_mci_dfs`, `mci_approx_dfs`), used only by the tests as a reference.

## 1. Exact MCI with Glynn's identity
Ring of n atoms, m = n - 1, AOMs S_0..S_m (q x q, q = occupied MOs or the natural orbitals). The raw sum is

    R = sum over permutations s of 1..m of  Tr(S_0 S_s1 ... S_sm)
      = 2^-(m-1) sum_{d in {+-1}^m, d_1 = +1} (prod_i d_i) Tr(S_0 X^m),   X = sum_i d_i S_i

Expanding X^m gives every word of length m; the sign sum cancels any word in which an atom appears an even number of
times, so only permutations survive. 2^(n-2) independent terms, memory independent of n (one batch per process).
MCI = prefactor * R / 2 (prefactor 2^(n-1) restricted, 2^(n/2) natural orbitals), as in the DFS code.

Evaluation of one term (`_kernel_glynn_krylov`): factor S_0 = U diag(w) V^T with its rank r0 (~ basis functions on the atom), then

    Tr(S_0 X^m) = sum_a w_a v_a^T X^m u_a
    symmetric AOMs:      u^T X^m u = (X^h u) . (X^(m-h) u)   -> ceil(m/2) products of X (q x q) with a q x r0 block
    Mulliken:            same with P = X^2 computed once

- The atom of lowest rank is put first.
- Matrices are projected onto the joint column space of the ring AOMs. This is exact and helps when the ring is part of a larger molecule.
- Natural orbitals: occ^(1/2) A is rewritten as occ^(1/4) A occ^(1/4). This is a similarity, so traces are unchanged, and the matrices become symmetric.
- Falls back to repeated squaring (`_kernel_glynn_exact`) when the block is not thin.

Proven limits (derivation panel, refereed):
- 2^(n-2) terms is optimal for any formula built from powers of sign-weighted sums (Waring rank of x_1...x_m).
- Any memory-free sum of products of linear forms could save at most ~sqrt(pi m / 8).
- R is a Hamiltonian-cycle polynomial (#P-hard).
- log-det, eigenvalue, Ryser and roots-of-unity forms are slower and less accurate.
- Splitting into halves always loses.

## 2. The approximations on an adjacency matrix (same engine for every mcialg)
Each algorithm becomes a 0/1 adjacency A built from the same BFS distances as the DFS:

| algorithm | allowed step u -> v |
|---|---|
| mcialg=1 | dist <= d |
| mcialg=3 / 5 | dist <= d and dist odd (5: shortest odd walk distances) |
| mcialg=2 / 4 | (all steps <= d) minus (all steps <= d-1); 4 with the odd-walk distances |
| mcialg=6 | DI(u, v) > delta |

Restricted Glynn identity: walks with a repeated atom cancel exactly as before, so

    R_A = 2^-(m-1) sum_d (prod d) F_A(d),
    F_A(d) = sum over closed walks 0 -> v1 -> ... -> vm -> 0 with every step allowed by A of  prod_t d_vt  Tr(S_0 S_v1 ... S_vm)

F_A is evaluated by a transfer recursion along the walk:

    B_1[v] = A[0,v] d_v S_0 S_v,   B_(t+1)[v] = d_v sum_u A[u,v] B_t[u] S_v,   F = sum_v A[v,0] Tr(B_m[v])

Low-rank form (`_kernel_glynn_lowrank`):
- S_v = U_v diag(w_v) V_v^T and G_uv = A[u,v] V_u^T U_v (r x r blocks), so the state is r0 x (m r) instead of m matrices of q x q.
- **Reachability cone:** at step t only atoms reachable from atom 0 in t steps that can still return in m - t steps are kept, so a sparse A gives smaller products.
- **Meet in the middle:** for symmetric AOMs and A, two half walks are joined, which halves the steps.

Number of permutations: the same recursion with weights 1, in exact integer arithmetic modulo primes (CRT), so it is
compared with the DFS count exactly.

Two-sided (bipartite) graphs (`_two_sides`):
- Detected from A, not assumed. If every allowed step joins two sides of equal size, every closed walk visits each side n/2 times.
- A second sign can then be fixed: 2^(n-3) terms (half). The cone keeps only one side per step.
- This is what mcialg=3/5 at any d gives on even alternant rings, so they use it automatically.
- Odd rings and non-alternant fused rings fall back to 2^(n-2) terms.
- Unequal sides mean no Hamiltonian cycle (sum = 0); the shortcut was wrongly applied there (adamantane), now fixed.

Sparse adjacency (`_pruned_sum`, mean degree <= 4, e.g. d = 1 or 2, fullerene cages):
- Glynn always needs 2^(n-2) terms however sparse A is, so these cases use a cycle search instead.
- The search drops a path as soon as an unused atom can no longer be entered and left.
- Cost scales with the number of cycles, not with n. C60 d=1: 1090 cycles in 10 s (workbook: ~1700 s on 8 cores).

Profiles (`mci_profile_glynn`): the mcialg=2/4 shells for all d in one pass (C_d - C_(d-1)); they add up to the exact MCI.

Dispatcher (`mci_approx_glynn` -> `_glynn_sum`):

| adjacency | method |
|---|---|
| complete | exact kernel |
| sparse | pruned search |
| two equal sides | 2^(n-3) terms |
| otherwise | low-rank or full-matrix walk kernel, cheapest by flop count |

## 3. Validation
- 900 checks against explicit enumeration (n = 3-8; symmetric, indefinite and non-symmetric matrices; ring, odd-only,
  random, directed and two-sided adjacencies): 0 mismatches, worst relative error 3e-13 (`check_brute.py`).
- 979 comparisons with the DFS on the test systems, mcialg 0-6 at every d: all PASS (`check_dfs.py`).
- Your dataset (IAO, from your .fchk, `run_dataset.py`):
  - 386 Glynn-vs-DFS comparisons.
  - The only failures were adamantane (bug above, fixed); its rows are still to be recomputed.
  - Three triplet .fchk files are empty (indole_t, benzothiophene_t, c10h10_t).
  - The run stopped at C50 (fullerenes); it is resumable.
- Old vs new kernels on the same real AOMs: agreement within 1e-9. mcialg=4 profiles sum to the exact MCI.
- `compare_small.py`: C6-C12 annulenes, every algorithm, DFS vs Glynn value / evaluations / time (DFS exact C12: 595 s, Glynn 0.02 s).

## 4. The 4n "failure" was the wavefunctions
- Your c16h16 has all bonds equal (1.418 A) and a 2.25 eV gap (c8h8/c12h12: alternating bonds, 4.4-5.0 eV).
- My first n=20/24 test rings had converged to a bad SCF solution.
- Rebuilt and stability-checked, every ring n = 8-24 gives mcialg=5 at kmax within 0.13 % of the exact value, for both equal and alternating bonds (`annulene_geom.py`).

## 5. Annulene scan (`scan_alg5.py`, RHF/STO-3G followed to a stable solution, meta-Lowdin, 8 cores)
| n | geometry | mcialg=5 kmax | exact | mcialg=5 / exact |
|---|---|---|---|---|
| 6 | equal | 0.005 s | 0.003 s | 99.998 % |
| 8 | alternating | 0.02 s | 0.008 s | 99.874 % |
| 10 | equal | 0.02 s | 0.01 s | 99.9993 % |
| 12 | alternating | 0.03 s | 0.05 s | 99.991 % |
| 14 | equal | 0.09 s | 0.23 s | 99.99998 % |
| 16 | alternating | 0.35 s | 0.73 s | 99.9997 % |
| 18 | equal | 0.61 s | 3.4 s | 99.9999996 % |
| 20 | alternating | 2.7 s | 16 s | 99.999992 % |
| 22 | equal | 13 s | 76 s | 99.99999998 % |
| 24 | alternating | 63 s | 412 s | 100.00003 % (exact has ~2e-13 rounding on a 5.6e-7 MCI) |
| 26 | equal | 313 s | 1938 s | 99.9999998 % |
| 28 | alternating | 1470 s | not run | - |

Growth: about x4.7 per two atoms for both. Extrapolated on 8 cores:
- mcialg=5 at kmax: n=30 ~2 h, 32 ~9 h, 34 ~2 days.
- exact: n=28 ~2.5 h, 30 ~12 h, 32 ~2.5 days.

## 6. What is not general yet / next steps
- Odd rings and non-bipartite fused systems use the general 2^(n-2) path: correct, but without the halving.
- Mulliken does not get the symmetric half-step saving.
- Pending:
  - choose DFS / pruned search / Glynn automatically by default
  - recompute the adamantane rows and finish the dataset (fullerenes from C50, surfaces)
  - build MCI_Approximation_Summary_Glynn.xlsx (`build_xlsx.py`)
  - rerun compare_small for C12
  - continue the scan from n=30
  - accuracy check of the exact sum at n >= 28 (rotation test)
  - full test suite, commit
