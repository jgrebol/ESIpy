import numpy as np
import os
import sys
sys.path.append(os.getcwd())
from esipy.readfchk import readfchk, read_list_from_fchk

def debug_dm_trace(fchk_path):
    print(f"\n--- Debugging DM Trace for: {fchk_path} ---")
    mol, mf = readfchk(fchk_path)
    n = mol.nao
    
    # 1. Get raw flat density from FCHK
    flat = read_list_from_fchk('Total SCF Density', fchk_path)
    mat = np.zeros((n, n))
    tril_idx = np.tril_indices(n)
    mat[tril_idx] = flat
    mat = mat + mat.T - np.diag(np.diag(mat))
    
    s = mol.intor_symmetric('int1e_ovlp')
    
    # No permutation
    pop_raw = np.trace(mat @ s)
    print(f"  Pop (Raw Matrix): {pop_raw:.6f}")
    
    # Gaussian permutation (what current dev-fchk does)
    from esipy.tools import permute_aos_rows
    mat_p = permute_aos_rows(mat, mol)
    mat_p = permute_aos_rows(mat_p.T, mol).T
    pop_p = np.trace(mat_p @ s)
    print(f"  Pop (Gau Permuted): {pop_p:.6f}")
    
    print(f"  Expected: {mol.nelectron}")

debug_dm_trace('../tests/FCHK/QCHEM/7_rmp2.fchk')
debug_dm_trace('../tests/FCHK/QCHEM/1_benzene_spherical.fchk')
