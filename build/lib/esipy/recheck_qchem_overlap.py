import numpy as np
from pyscf import gto, scf
from esipy.readfchk import readfchk
import os

def check():
    path = "../tests/FCHK/QCHEM/1_benzene_spherical.fchk"
    mol_f, mf_f = readfchk(path)
    S_f = mf_f.get_ovlp()
    
    # Standard PySCF Benzene
    basis = 'cc-pVTZ'
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
    """
    mol_p = gto.M(atom=atom, basis=basis).build()
    S_p = mol_p.intor('int1e_ovlp')
    
    # Try just the diagonals first
    diff_diag = np.max(np.abs(np.diag(S_f) - np.diag(S_p)))
    print(f"Max Diff Diag: {diff_diag:.2e}")
    
    # If diagonals match, maybe off-diagonals are just small differences?
    print(f"Max Diff Full: {np.max(np.abs(S_f - S_p)):.2e}")
    
    # What if we look at the first 10x10?
    print("\nPySCF S (10x10):\n", S_p[:10,:10])
    print("\nQChem S (10x10):\n", S_f[:10,:10])

check()
