import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))
from esipy.readfchk import readfchk
from esipy.iao import iao
import numpy as np

path = '../joan/LiH/GS/lih_1.0.fchk'
mol, mf, myhf = readfchk(path)
S = mol.intor('int1e_ovlp')

occ_vals = mf.mo_occ
idx_all = occ_vals > 1e-10
c_all = mf.mo_coeff[:, idx_all]
occ_all = occ_vals[idx_all]

idx_main = occ_vals > 0.5
c_main = mf.mo_coeff[:, idx_main]

print(f"Number of main NOs (occ > 0.5): {c_main.shape[1]}")
print(f"Number of all NOs (occ > 1e-10): {c_all.shape[1]}")

# Case 1: Build IAO from all NOs
U_all_nonorth, pmol = iao(mol, c_all)
U_all = S @ U_all_nonorth
eta = [np.zeros((pmol.nao, pmol.nao)) for _ in range(pmol.natm)]
for i in range(pmol.natm):
    start, end = pmol.aoslice_by_atom()[i, -2], pmol.aoslice_by_atom()[i, -1]
    eta[i][start:end, start:end] = np.eye(end - start)

pop0_all = np.trace(np.diag(occ_all) @ c_all.T @ U_all @ eta[0] @ U_all.T @ c_all)
pop1_all = np.trace(np.diag(occ_all) @ c_all.T @ U_all @ eta[1] @ U_all.T @ c_all)
print(f"IAO (all NOs) Populations: {pop0_all:.4f}, {pop1_all:.4f}, Total: {pop0_all+pop1_all:.4f}")

# Case 2: Build IAO from main NOs only, but build AOMs with all NOs
U_main_nonorth, pmol = iao(mol, c_main)
U_main = S @ U_main_nonorth
pop0_main = np.trace(np.diag(occ_all) @ c_all.T @ U_main @ eta[0] @ U_main.T @ c_all)
pop1_main = np.trace(np.diag(occ_all) @ c_all.T @ U_main @ eta[1] @ U_main.T @ c_all)
print(f"IAO (main NOs for U, all NOs for AOM) Populations: {pop0_main:.4f}, {pop1_main:.4f}, Total: {pop0_main+pop1_main:.4f}")
