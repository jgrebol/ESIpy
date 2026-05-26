import numpy as np
import os
import sys
sys.path.append(os.getcwd())
from esipy.readfchk import readfchk

def test_permutations(filename):
    path = os.path.join('../tests/FCHK/QCHEM', filename)
    if not os.path.exists(path): return
    
    print(f"\n=== Testing Permutations for {filename} ===")
    from esipy.readfchk import read_list_from_fchk
    
    # 1. Get raw density
    mol, mf = readfchk(path)
    n = mol.nao
    flat = read_list_from_fchk('Total SCF Density', path)
    mat_gau = np.zeros((n, n))
    tril_idx = np.tril_indices(n)
    mat_gau[tril_idx] = flat
    mat_gau = mat_gau + mat_gau.T - np.diag(np.diag(mat_gau))
    
    s = mol.intor_symmetric('int1e_ovlp')
    
    # Try no permutation (Current dev-fchk)
    pop_none = np.trace(mat_gau @ s)
    print(f"  Pop (No Permutation): {pop_none:.6f}")
    
    # Try Gaussian permutation
    from esipy.tools import permute_aos_rows
    mat_pyscf = permute_aos_rows(mat_gau, mol)
    mat_pyscf = permute_aos_rows(mat_pyscf.T, mol).T
    pop_gau = np.trace(mat_pyscf @ s)
    print(f"  Pop (Gau Permutation): {pop_gau:.6f}")

    # Check which one is closer to mol.nelectron
    print(f"  Expected: {mol.nelectron}")

test_permutations('2_benzene_cartesian.fchk')
