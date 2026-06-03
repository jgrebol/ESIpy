import numpy as np
from pyscf import gto, scf
from esipy.readfchk import readfchk
import os

def check_qchem_raw():
    path = "../tests/FCHK/QCHEM/1_benzene_spherical.fchk"
    mol_f, mf_f = readfchk(path)
    
    # Let's rebuild PySCF mol exactly
    basis = 'cc-pVTZ'
    atom = """
    C        0.000000000      0.000000000      1.393096000
    C        0.000000000      1.206457000      0.696548000
    C        0.000000000      1.206457000     -0.696548000
    C        0.000000000      0.000000000     -1.393096000
    C        0.000000000     -1.206457000     -0.696548000
    C        0.000000000     -1.206457000      0.696548000
    1        0.000000000      0.000000000      2.483127000
    1        0.000000000      2.150450000      1.241569000
    1        0.000000000      2.150450000     -1.241569000
    1        0.000000000      0.000000000     -2.483127000
    1        0.000000000     -2.150450000     -1.241569000
    1        0.000000000     -2.150450000      1.241569000
    """
    mol_p = gto.M(atom=atom, basis=basis).build()
    S_p = mol_p.intor('int1e_ovlp')
    S_f = mf_f.get_ovlp()
    
    print(f"Q-Chem S_f shape: {S_f.shape}")
    print(f"PySCF  S_p shape: {S_p.shape}")
    
    # Try all permutations of the atoms to see if it's just atom ordering?
    # Q-Chem FCHKs usually preserve atom order. 
    # Let's check the FIRST ATOM'S submatrix (Carbon 1, cc-pVTZ: 14 AOs)
    s1_f = S_f[:14, :14]
    s1_p = S_p[:14, :14]
    
    print("\nFirst Atom (C) Overlap Submatrix Comparison:")
    print(f"  Max Diff: {np.max(np.abs(s1_f - s1_p)):.2e}")
    
    # Try just pure permutation of the AOs within the atom
    from itertools import permutations
    # C in cc-pVTZ has s,p,d,f functions. 
    # s: 3, p: 2 (6 AOs), d: 1 (5 AOs) ? 
    # 1s, 2s, 3s, 2p, 3p, 3d, 4f...
    # Let's check diag
    print(f"  Diag (FCHK): {np.diag(s1_f)}")
    print(f"  Diag (PyST): {np.diag(s1_p)}")

check_qchem_raw()
