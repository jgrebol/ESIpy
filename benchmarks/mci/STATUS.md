# MCI beyond the DFS: status (handoff file, update as work proceeds)

Branch dev-mciaprox.

## Done
- `esipy/mci.py`: Glynn exact (`compute_mci_glynn`), restricted sums on the approximations' adjacency
  (`mci_approx_glynn`, K-space and low-rank walk kernels), exact permutation counts, `method='glynn'` switch.
  DFS untouched and default. `tests/test13_glynn_mci.py` passes. 856 DFS-vs-Glynn checks, 0 mismatches.
- `benchmarks/mci/systems.py` + `cache/systems.pkl`: annulenes 6-24 (STO-3G), 10-18 (def2-SVP), acenes, UHF, CASSCF.
- Derivation panel (4 angles, every proposal refereed). Scripts in `derivation/<angle>/`.
  Prototype of the Krylov kernel: `derivation/main/krylov_proto.py`.

## Verified findings to implement
1. Krylov-Glynn (exact): Tr(S0 X^m) = sum_a w_a (X^h u_a).(X^(m-h) u_a), S0 = U diag(w) U^T (eigh, keep BOTH signs of w),
   ceil(m/2) products q x q by q x r0. Non-symmetric (Mulliken): X2 = X.X once, then ceil(m/2) block products.
   Fix the lowest-rank atom. Symmetry test must be strict (no rtol). Natural orbitals: symmetrise with occ^(1/4).
   Fall back to repeated squaring when ceil(m/2) r0 >~ 5 q. Measured x2.5-5 in numpy (flops x5-10).
2. Restricted sums: meet-in-the-middle (floor(m/2) products with T) for symmetric AOMs + symmetric adjacency.
3. Bipartite adjacency (alg 3/5 on alternant rings, any d): fix one sign per side -> 2^(n-3) terms; do NOT add self-loops.
   Alternating-side state (h r columns instead of m r): x7 measured at n=16-18.
4. alg 2/4 profile: telescoping C_d - C_(d-1), reuse cumulative sums across d (kmax runs, not 2 kmax); last one is the exact MCI.
5. Dispatcher: DFS only when the cycle count is small (d <= 2 on simple rings); Glynn already wins at d = 3.

## Proven dead ends (do not retry)
- Fewer than 2^(m-1) terms for the exact sum with power-sum formulas (Waring rank of x1...xm); at most ~sqrt(pi m/8)
  for any sum of products of linear forms (Nisan), not attained at m=3,4. Splitting into halves always loses.
- log-det / eigenvalue / Ryser / roots of unity: slower and less accurate. Sum rule of the AOMs: no gain.
- Complement inclusion-exclusion loses beyond ~2 missing edges. Block-sparse transfer: no gain in numpy.
- The raw sum is a Hamiltonian-cycle polynomial (#P-hard): 2^n is the floor.

## Missing
- Implement 1-5 in `esipy/mci.py`, tests against DFS/brute force, full test suite.
- Dead-end pruning in the DFS fallback (C60 Isphere: 1090 cycles found in 5 s in pure Python).
- Permutation-count formulas (agent stopped by the usage limit; partial scripts in `derivation/counts/`).
- Benchmark on all molecules of joan/MCIAPROX/BENCHMARK (GEOMS, NEWGEOMS) + synthetic AOM-like matrices
  (q = occupied MOs, rank = basis functions per atom) for large n; state the largest ring per method.
- Machine constraints: 15 GB RAM, has crashed under load. One job at a time, OPENBLAS_NUM_THREADS=1, at most
  ~4 processes, no filesystem-wide find, keep PySCF runs small (or use synthetic matrices).

## Update: merged-path (frontier) sum, prototype in derivation/main/frontier.py
- Walk the adjacency position by position, merge partial paths with the same (used set, last atom), low-rank blocks
  r0 x r per state, dead-state pruning. Exact (1e-16 vs Glynn, same counts), no sign cancellation.
- States grow with n (not constant): alg1 d=3: 6160 (n=14), 51648 (n=18); alg5 d=kmax: 2800 (n=14), 44100 (n=18).
  Fast (n=18 alg5 d=7: 1.7 s vs ~19 s current Glynn) but MEMORY grows ~2^n: fine to n~22-24, not beyond. Needs a budget + fallback to Glynn.
- Distance truncation converges slowly: n=14 alg5 d=1/3/5/7 = 18/46/71/100 % of exact; n=18 d=7 = 74 %. Full d=kmax is needed.
- ALWAYS run with `ulimit -v 3000000` (machine crashed from a multi-GB intermediate array).

## Update: new Glynn kernels are IN esipy/mci.py (uncommitted)
- `_kernel_glynn_krylov` (exact, thin-block products), `_kernel_glynn_lowrank` rewritten (reachability cone per step,
  meet-in-the-middle for symmetric AOMs), `_two_sides` (second fixed sign when the adjacency is two-coloured, found
  from the matrix), lowest-rank atom fixed first. Old kernels kept as fall-backs.
- Checks: benchmarks/mci/check_brute.py (720 brute-force checks, 0 mismatches, worst 3e-13);
  check_kernels.py (old vs new on real AOMs, <= 1e-9); tests/test13 + test10 pass.
- Single core, meta-Lowdin annulenes (time_glynn.py): exact n=14/16/18/20 = 0.10/0.55/3.1/16.5 s (was 0.7/5.7/42/130);
  alg5 kmax = 0.02/0.12/0.56/2.7 s (was 0.35/1.8/19/58). Mulliken ~1.3x (exact) and ~3x (alg5) slower than symmetric.
- n=16/20 values differ from the first session because the SCF was rebuilt (4n rings), not because of the kernels.
- Still missing: full test suite, benchmark on all molecules + synthetic matrices, n=32 run, accuracy vs n
  (rotation test), count formulas, natural orbitals symmetrised with occ^(1/4), commit.

## Update: dataset validation in progress
- `esipy/mci.py` now also has: `_pruned_sum` (sparse adjacency, used automatically when mean degree <= 4; C60 d=1: 1090 cycles
  in 10 s), `mci_profile_glynn` (mcialg 2/4 shells in one go), `_prep_matrices_sym` (natural orbitals symmetrised).
- check_dfs.py: 979 DFS-vs-Glynn comparisons, all algorithms PASS. check_pruned.py: 0 mismatches.
- dataset.py (AOMs from your .fchk, IAO, cached), run_dataset.py (resumable, appends dataset_results.jsonl),
  build_xlsx.py (writes MCI_Approximation_Summary_Glynn.xlsx, original untouched), large_rings.py (n=20-26).
- Workbook values = ESIpy values / 2^(n-1). c14h14 alg5 perms should be 1,814,400 (= 7! 6!/2), workbook has 1,774,080.
- mcialg=5 at kmax vs exact: ~100 % for 4n+2 rings, but c16h16 -64 %, synthetic n=20 -92 %, n=24 -98 %: fails for large 4n rings.
- To do: finish DATASET_S/T, FULLERENES, SURFACES; build xlsx; n=26; then larger systems.

## Update
- BUG FIXED: `_two_sides` fixed a second sign also when the two sides had unequal size (adamantane, alg 3/5 d=4:
  Glynn 1200 perms, DFS 0). Now requires equal sides. check_brute.py has two-sided graphs: 900 checks, 0 mismatches.
- Dataset run (run_dataset.py): 386 comparisons with the DFS, only the adamantane rows failed (that bug); recompute them
  (delete their lines in dataset_results.jsonl and rerun the folders, the script skips finished keys).
- DATASET_T indole_t, benzothiophene_t, c10h10_t: the .fchk files are EMPTY (0 bytes) in the user's folder.
- large_rings.jsonl: n=20,22,24 done (exact, alg5 kmax, alg4 profile; profiles sum to the exact MCI).
  n=26 missing: annulene26 is not in cache/systems.pkl (build it, STO-3G is cheap).
- compare_small.py: DFS vs Glynn for c6h6, c8h8, c10h10, c12h12, all algorithms, values/evaluations/times -> compare_small.log.
  DFS exact c12h12: 595 s, Glynn 0.02 s.

## Update: the 4n failure was the wavefunctions, not mcialg=5
- Dataset c16h16: all C-C bonds equal (1.418 A, D16h), gap 2.25 eV (c8h8/c12h12 alternate 1.34/1.48, gap 4.4-5.0 eV).
- cache/systems.pkl annulene20/24 (STO-3G) had converged to a bad SCF: rebuilt from scratch (annulene_geom.py), n=20 gives
  mcialg=5 = 100.0000 % of exact for equal AND alternating bonds; n=8/12/16 99.87-100.03 %. DO NOT use systems.pkl n>=20.
- large_rings.jsonl percentages for n=20/24 are therefore invalid (timings are fine).
- Running: scan_alg5.py (annulenes 6..40, 4n+2 equal / 4n alternating, RHF/STO-3G with stability following, exact while
  n<=26, alg5 until a ring is predicted > 30 h), output scan_alg5.jsonl + scan_alg5.log, 8 cores.
- To redo later: compare_small.py for c12h12 only (cut by its 25-min timeout after alg1 d=5).
- User: CPU free (8-12 cores, do not fill), watch RAM (fullerene dataset processes ~2.2 GB each).

## Resume (after restart)
- Detached jobs: `resume.sh` (dataset, skips finished rings, then build_xlsx.py; log dataset_run.log, ends with DATASET_DONE)
  and `resume_scan.sh` (scan from n=30, 6 cores; log scan_alg5.log, ends with SCAN_DONE). Rerun either script after a crash.
- Adamantane rows removed from dataset_results.jsonl (backup dataset_results.bak) so they are recomputed with the fix.

## mcialg=5 maximum distance (fix)
- run_dataset.py used the plain BFS kmax for mcialg=5; it must be the max of the ODD-walk distances (kmax_odd), larger on
  odd / non-alternant rings. Fixed for new rows. Rows written before (ANNULENES, DATASET_S, DATASET_T) need an alg5 pass at
  kmax_odd (only changes non-bipartite rings).
- Finding: on non-bipartite rings mcialg=5 at kmax_odd keeps ALL permutations (= exact, no saving; furan, c7h7+, pentalene,
  heptalene, acenaphthylene, benzofuran: 100 %). It is a real approximation only on bipartite (alternant even) rings:
  closed-shell singlets ~99.9-100 % (annulenes to n=26), distorted c10h10_twist 90 %, c8h8_boat 96 %; triplets poor
  (benzene_t 129 %, naphthalene_t -13 %).

## Limits run + plots
- `limits.sh` (detached): waits for alg5 n=30, then exact n=28 -> alg5 n=32 -> exact n=30 (8 cores, one at a time,
  outputs scan_alg5.jsonl / scan_exact.jsonl, LIMITS_DONE in scan_alg5.log). Rerun limits.sh after a crash (steps resume).
- `plot_annulenes.py` -> plots/annulenes_mci.{png,pdf}; rerun when rings are added.
- Fullerenes C20-C52: cycle counts = workbook, Glynn = DFS on same AOMs; normalised MCI = workbook up to C48,
  C50/C52 differ in 3rd decimal (AOMs differ from the workbook run). Pruned search C44 0.9 s vs 3698 s (workbook, 8 cores).
- after_dataset.sh waits for DATASET_DONE, then fix_alg5.py (mcialg=5 at kmax_odd for old rows) and build_xlsx.py.

## Simplification (user request: minimal ESIpy changes, DFS only as legacy, no options)
- esipy/mci.py = original file + renames compute_mci -> compute_mci_dfs, mci_approx -> mci_approx_dfs (legacy, untouched),
  natural orbitals as occ^(1/4) A occ^(1/4) in _prep_matrices, fsum import, and one Glynn section with the public
  compute_mci / mci_approx (same signatures, max_mem = MB per batch) and mci_profile. No `method` option. __init__.py reverted.
- Removed kernels: matrix-power exact and full-matrix walks (Krylov and low-rank kernels are general).
- Glynn terms summed with math.fsum. Full previous version kept in scratchpad/mci_full_before_simplify.py.
- All benchmark scripts and tests/test13 updated to the new names (legacy DFS via *_dfs).
- Checks after simplifying: check_brute 900/0, check_pruned 0 mismatches, test13 + test10 pass.
- Running: run_tests.sh (full suite on the branch; baseline dev-mciaprox: 96 passed, 2 skipped), then run_accuracy.sh
  (rotation test n=12..24, accuracy.jsonl). Review agent on the Glynn section; counts agent resumed.
- Full test suite on the simplified branch: 96 passed, 2 skipped (same as the dev-mciaprox baseline). Adamantane rows recomputed: match DFS.
- Accuracy (rotations): exact relative spread 1.7e-12 (n=12), 3.7e-11 (16), 3.6e-9 (20, MCI 5.8e-6, abs 2e-14);
  mcialg=5 stays ~1e-14 relative. The exact sum loses digits on 4n rings with tiny MCI; check n=22/24 in accuracy.log.
