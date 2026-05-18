import sys
import os
import numpy as np
# Add the parent directory to sys.path to allow imports from esipy
sys.path.append(os.path.dirname(os.getcwd()))

from esipy.readfchk import readfchk
from pyscf import scf

path = '../joan/LiH/GS/lih_1.0.fchk'
print(f"Testing coefficients in: {path}")
mol, mf, _ = readfchk(path) # mf currently gets the FCHK MOs
c_fchk = mf.mo_coeff

# 1. Compute HF Energy using FCHK Coefficients
mf_test = scf.RHF(mol.pyscf_mol)
h1 = mf_test.get_hcore()
s = mf_test.get_ovlp()
# Density from first 2 FCHK MOs (assuming they are HF-like)
d_fchk = c_fchk[:, :2] @ c_fchk[:, :2].T * 2
vhf = mf_test.get_veff(dm=d_fchk)
e_hf_fchk = np.einsum('ij,ji->', h1 + 0.5*vhf, d_fchk) + mol.pyscf_mol.energy_nuc()
print(f"HF Energy using FCHK coefficients: {e_hf_fchk:.8f}")

# 2. Compute Converged PySCF HF Energy
mf_pyscf = scf.RHF(mol.pyscf_mol)
mf_pyscf.verbose = 0
e_pyscf = mf_pyscf.kernel()
print(f"HF Energy using converged PySCF:   {e_pyscf:.8f}")

print(f"Difference: {abs(e_hf_fchk - e_pyscf):.2e}")
