import sys
import os
import numpy as np
# Add the parent directory to sys.path to allow imports from esipy
sys.path.append(os.path.dirname(os.getcwd()))

from esipy.readfchk import readfchk
from esipy.make_aoms import make_aoms
from pyscf import scf

path = '../joan/OUISSAM/lih.fchk'
print(f"Testing with HF FCHK: {path}")
mol, mf, _ = readfchk(path) 

print(f"Primary mf name: {mf.__name__}")

# Case A: Using coefficients directly from FCHK
print("\nCase A: Using FCHK coefficients")
aom_fchk = make_aoms(mol, mf, 'mulliken')
pop_fchk = [2.0 * np.trace(m) for m in aom_fchk]
print(f"Populations (FCHK): Li {pop_fchk[0]:.6f}, H {pop_fchk[1]:.6f}")

# Case B: Using coefficients from a new PySCF calculation
print("\nCase B: Using new PySCF RHF calculation")
# Use the underlying PySCF mole object
mf_pyscf = scf.RHF(mol.pyscf_mol)
mf_pyscf.verbose = 0
mf_pyscf.kernel()
aom_pyscf = make_aoms(mol.pyscf_mol, mf_pyscf, 'mulliken')
pop_pyscf = [2.0 * np.trace(m) for m in aom_pyscf]
print(f"Populations (PySCF): Li {pop_pyscf[0]:.6f}, H {pop_pyscf[1]:.6f}")

diff = np.abs(np.array(pop_fchk) - np.array(pop_pyscf)).max()
print(f"\nMax population difference: {diff:.2e}")
