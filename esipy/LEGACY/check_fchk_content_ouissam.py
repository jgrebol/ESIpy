import sys
import os
import numpy as np
# Add the parent directory to sys.path to allow imports from esipy
sys.path.append(os.path.dirname(os.getcwd()))

from esipy.readfchk import readfchk, read_list_from_fchk, permute_aos_rows

path = '../joan/OUISSAM/lih.fchk'
mol, mf, _ = readfchk(path) 
S = mol.intor('int1e_ovlp')

# 1. Read Raw Alpha MO Coefficients from FCHK
c_fchk = mf.mo_coeff 

# 2. Read Total SCF Density from FCHK
dens_flat = read_list_from_fchk('Total SCF Density', path)
n = mol.nao
d_ao = np.zeros((n, n))
d_ao[np.tril_indices(n)] = np.array(dens_flat, dtype=float)
d_ao = d_ao + d_ao.T - np.diag(d_ao.diagonal())
d_ao = permute_aos_rows(d_ao, mol)
d_ao = permute_aos_rows(d_ao.T, mol).T

# 3. Project Density onto MO basis
d_mo = c_fchk.T @ S @ d_ao @ S @ c_fchk

print("Diagonal of Density in MO basis:")
diag = np.diag(d_mo)
print(diag[:10])

is_diagonal = np.abs(d_mo - np.diag(diag)).max() < 1e-4
print(f"Is Density diagonal in MO basis? {is_diagonal}")

is_hf_density = np.all((np.abs(diag - 2.0) < 1e-4) | (np.abs(diag - 0.0) < 1e-4))
print(f"Is it an HF-like density (integer occupations)? {is_hf_density}")
