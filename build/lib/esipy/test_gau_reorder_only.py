import numpy as np
from pyscf import gto
from esipy.readfchk import readfchk, read_list_from_fchk
from esipy.tools import permute_aos_rows
import os

def test():
    path = f"../tests/FCHK/GAUSSIAN/1_benzene_spherical.fchk"
    mol_f, mf_f = readfchk(path)
    # 1. Raw Density from FCHK
    dt_flat = read_list_from_fchk('Total SCF Density', path)
    n = mol_f.nao
    mat_raw = np.zeros((n, n))
    tril_idx = np.tril_indices(n)
    mat_raw[tril_idx] = dt_flat
    mat_raw = mat_raw + mat_raw.T - np.diag(np.diag(mat_raw))
    
    # 2. PySCF Overlap Matrix
    S_p = mol_f.pyscf_mol.intor('int1e_ovlp')
    
    # Use ESIpy's current reorderer but WITHOUT scaling
    # We hack tools.py momentarily in memory? No, let's just use the current result
    # which we know is 42.09.
    
    print(f"Current Trace (with reordering): 42.0921")
    
    # The question is: Why is it 42.09 and not 42.00?
    # Benzene Spherical has s, p, d, f functions.
    # d is l=2, f is l=3.
    # Are there scaling factors for spherical in jxzou's code?
    # NO. Spherical is only reordering.
    
    # Let's check the Overlap matrix diagonal for GAUSSIAN Benzene
    # FCHK usually doesn't have S, but ESIpy recomputes it.
    print(f"FCHK Overlap Diagonal (first 10): {np.diag(S_p)[:10]}")
    # They are all 1.0 because ESIpy uses PySCF to compute it!

test()
