import numpy as np
from pyscf import gto, dft
from esipy import ESI
from esipy.tools import find_di

# Match Gaussian B3LYP exactly
dft.libxc.B3LYP_WITH_VWN5 = True

def run_bs():
    # H2 at 3.0 Angstroms
    mol = gto.M(atom="H 0 0 0; H 0 0 3.0", basis='6-31G(d)', spin=0).build()
    
    # Unrestricted Singlet (UKS)
    mf = dft.UKS(mol).set(xc='B3LYP')
    
    # 1. Create a "localized" initial guess
    # Standard guess is restricted. We need to manually break it.
    dm = mf.get_init_guess()
    # dm has shape (2, nao, nao)
    # Put all alpha on atom 0, all beta on atom 1
    # Index 0 is H1 orbital, Index 1 is H2 orbital (roughly)
    dm[0, 0, 0] = 1.0
    dm[0, 1, 1] = 0.0
    dm[1, 0, 0] = 0.0
    dm[1, 1, 1] = 1.0
    
    # 2. Add a small level shift to stabilize the BS state during optimization
    mf.level_shift = 0.2
    
    print("--- Running PySCF Broken Symmetry UKS ---")
    mf.kernel(dm)
    
    print(f"Final Energy: {mf.e_tot:.6f}")
    s2, mult = mf.spin_square()
    print(f"<S^2>: {s2:.4f}")
    
    # 3. Check Indicators
    esi = ESI(mol=mol, mf=mf, partition='mulliken')
    aoms = esi.aom
    di = find_di(aoms[0], 1, 2) + find_di(aoms[1], 1, 2)
    print(f"DI(1,2): {di:.6f}")

run_bs()
