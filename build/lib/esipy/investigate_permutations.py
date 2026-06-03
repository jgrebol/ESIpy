import numpy as np
from pyscf import gto, scf
from esipy.readfchk import readfchk, read_list_from_fchk
from esipy.tools import permute_aos_rows
import os

def check_raw_vs_permuted(prog, filename):
    path = f"../tests/FCHK/{prog}/{filename}"
    if not os.path.exists(path): return
    
    print(f"\n--- Investigating {prog}: {filename} ---")
    
    # 1. Get RAW DM from FCHK (before ESIpy touches it)
    dt_flat = read_list_from_fchk('Total SCF Density', path)
    if len(dt_flat) == 0: 
        dt_flat = read_list_from_fchk('Total MP2 Density', path)
    
    # Simple triangular rebuild (No reordering)
    mol_f, mf_f = readfchk(path)
    n = mol_f.nao
    mat_raw = np.zeros((n, n))
    tril_idx = np.tril_indices(n)
    mat_raw[tril_idx] = dt_flat
    mat_raw = mat_raw + mat_raw.T - np.diag(np.diag(mat_raw))
    
    # 2. Get PySCF native metric
    basis = 'cc-pVTZ' if 'benzene' in filename else 'cc-pVDZ'
    atom = """
    C        0.000000000      0.000000000      1.393096000
    C        0.000000000      1.206457000      0.696548000
    C        0.000000000      1.206457000     -0.696548000
    C        0.000000000      0.000000000     -1.393096000
    C        0.000000000     -1.206457000     -0.696548000
    C        0.000000000     -1.206457000      0.696548000
    H        0.000000000      0.000000000      2.483127000
    H        0.000000000      2.150450000      1.241569000
    H        0.000000000      2.150450000     -1.241569000
    H        0.000000000      0.000000000     -2.483127000
    H        0.000000000     -2.150450000     -1.241569000
    H        0.000000000     -2.150450000      1.241569000
    """ if 'benzene' in filename else "O 0.0 0.0 0.0; H 0.0 0.757 0.586; H 0.0 -0.757 0.586"
    mol_p = gto.M(atom=atom, basis=basis, cart=('cartesian' in filename)).build()
    S_p = mol_p.intor('int1e_ovlp')
    
    # 3. Test scenarios
    # A. Raw Trace
    print(f"  Trace(P_raw @ S_pyscf):      {np.trace(mat_raw @ S_p):.4f}")
    
    # B. ESIpy Reordered Trace
    mat_esi = permute_aos_rows(mat_raw, mol_f)
    mat_esi = permute_aos_rows(mat_esi.T, mol_f).T
    print(f"  Trace(P_esi @ S_pyscf):      {np.trace(mat_esi @ S_p):.4f}")
    
    # C. Reordered Trace WITHOUT scaling (Only permutation)
    # Mocking permute_aos_rows logic without SDIAG
    def permute_only(mat, mole2):
        from esipy.tools import build_mapping
        maps = {2: [0, 3, 4, 1, 5, 2], 3: [0, 4, 5, 3, 9, 6, 1, 8, 7, 2],
                -2: [4, 2, 0, 1, 3], -3: [6, 4, 2, 0, 1, 3, 5]}
        idx_list = []
        atom_map = np.asarray(mole2.fchk_basis_arrays['iatsh']) - 1
        shell_types = np.asarray(mole2.fchk_basis_arrays['mssh'])
        for i in range(len(shell_types)):
             l = shell_types[i]
             start = sum([abs(shell_types[k]) for k in range(i)]) # Simple approximation
             # We need real start indices
             pass 
        # Actually ESIpy's tools.py has permute_aos_rows, let's just hack it temporarily to skip SDIAG
        return mat # Placeholder for logic below
        
    if prog == 'QCHEM':
         # Just reordering, no scaling (Q-Chem assumption)
         # We need to find the correct map for Q-Chem
         pass

check_raw_vs_permuted('GAUSSIAN', '1_benzene_spherical.fchk')
check_raw_vs_permuted('QCHEM', '1_benzene_spherical.fchk')
