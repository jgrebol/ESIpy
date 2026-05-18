import sys
import os
import numpy as np
# Add the parent directory to sys.path to allow imports from esipy
sys.path.append(os.path.dirname(os.getcwd()))

from esipy.readfchk import readfchk
from pyscf import scf

path = '../joan/LiH/GS/lih_1.0.fchk'
print(f"Testing with CASSCF FCHK: {path}")
mol, mf, myhf = readfchk(path) # mf=CASSCF, myhf=RHF (recomputed)

# FCHK Coefficients (what I assigned to CAS MOs)
c_fchk = mf.mo_coeff

# PySCF RHF Coefficients
c_pyscf = myhf.mo_coeff

# Compare
# Note: they might have different signs/ordering
# We can compare the density they generate
d_fchk = c_fchk[:, :2] @ c_fchk[:, :2].T * 2
d_pyscf = c_pyscf[:, :2] @ c_pyscf[:, :2].T * 2

diff = np.abs(d_fchk - d_pyscf).max()
print(f"Max difference between FCHK MO density and PySCF HF density: {diff:.2e}")

# Check if FCHK coefficients diagonalize the FCHK density
S = mol.intor('int1e_ovlp')
dens_flat = mf.make_rdm1(ao_repr=True) # This is the FCHK density
d_mo = c_fchk.T @ S @ dens_flat @ S @ c_fchk
diag = np.diag(d_mo)
is_diagonal = np.abs(d_mo - np.diag(diag)).max() < 1e-4
print(f"Is FCHK density diagonal in FCHK MO basis? {is_diagonal}")
