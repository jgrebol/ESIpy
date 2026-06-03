import numpy as np
from pyscf import gto, scf
from esipy.readfchk import readfchk
import os

def mock_permute(mat, mole2, source='GAUSSIAN'):
    # Replicate the mapping logic from tools.py/Fortran
    MAPS = {
        2: [0, 3, 4, 1, 5, 2],         # 6D
        3: [0, 4, 5, 3, 9, 6, 1, 8, 7, 2], # 10F
        -2: [4, 2, 0, 1, 3],           # 5D
        -3: [6, 4, 2, 0, 1, 3, 5],     # 7F
    }
    
    # Gaussian Sdiag (approximate constants from Mokit/PySCF convention)
    # PySCF uses normalized-to-unity Cartesian basis functions.
    # Gaussian FCHK often stores them in a standard normalization where xx is normalized,
    # but xy might have a factor of sqrt(3) etc.
    SQRT3 = np.sqrt(3.0)
    SQRT5 = np.sqrt(5.0)
    
    SDIAG = {
        2: [1.0, SQRT3, SQRT3, 1.0, SQRT3, 1.0], # XX, XY, XZ, YY, YZ, ZZ ?
        # Actual mapping depends on the target order XX, XY, XZ, YY, YZ, ZZ
    }

    if source == 'QCHEM':
        # User hint: Q-Chem overlap is same as PySCF.
        # Let's check if the DM also follows this.
        return mat

    # For Gaussian, let's try just the permutation first (identity scaling)
    # (The existing ESIpy code applies SDIAG scaling)
    return mat

def check_consistency(prog, filename):
    path = f"../tests/FCHK/{prog}/{filename}"
    if not os.path.exists(path): return
    
    print(f"\n--- Checking {prog}: {filename} ---")
    mol_fchk, mf_fchk = readfchk(path)
    # The readfchk logic ALREADY calls permute_aos_rows inside MeanField2.rebuild()
    dm_rebuilt = mf_fchk.make_rdm1()
    
    # Let's get the RAW matrix before permutation (we have to reach into readfchk internals or mock it)
    # Instead, let's just use the current result and see how far it is from PySCF.
    
    if 'benzene' in filename:
        basis = 'cc-pVTZ'; atom = """
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
        mol_p = gto.M(atom=atom, basis=basis, cart=('cartesian' in filename)).build()
        mf_p = scf.RHF(mol_p).run()
        dm_p = mf_p.make_rdm1()
    else: return

    S_p = mol_p.intor('int1e_ovlp')
    S_f = mf_fchk.get_ovlp()
    
    print(f"  P_rebuilt @ S_pyscf Trace: {np.trace(dm_rebuilt @ S_p):.4f}")
    print(f"  P_rebuilt @ S_fchk  Trace: {np.trace(dm_rebuilt @ S_f):.4f}")
    
    if prog == 'QCHEM':
        # Check if S_fchk is close to S_p
        diff_S = np.max(np.abs(S_f - S_p))
        print(f"  Max Diff Overlap (Q-Chem vs PySCF): {diff_S:.2e}")
        if diff_S < 1e-6:
             print("  SUCCESS: Q-Chem Overlap matches PySCF exactly.")
        else:
             print("  FAILURE: Q-Chem Overlap does NOT match PySCF order.")

check_consistency('GAUSSIAN', '1_benzene_spherical.fchk')
check_consistency('QCHEM', '1_benzene_spherical.fchk')
