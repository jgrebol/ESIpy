import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))
from esipy.readfchk import readfchk
from esipy.make_aoms import make_aoms
import numpy as np

path = '../joan/LiH/GS/lih_1.0.fchk'
mol, mf, myhf = readfchk(path)

# Custom make_aoms-like logic for testing
S = mol.intor('int1e_ovlp')
eta = [np.zeros((mol.nao, mol.nao)) for _ in range(mol.natm)]
for i in range(mol.natm):
    start, end = mol.aoslice_by_atom()[i, -2], mol.aoslice_by_atom()[i, -1]
    eta[i][start:end, start:end] = np.eye(end - start)

occ = mf.mo_occ
c_no = mf.mo_coeff
c_hf = myhf.mo_coeff

print("Mulliken with NO basis:")
aoms_no = [c_no.T @ S @ eta[i] @ c_no for i in range(mol.natm)]
pop0_no = np.trace(np.diag(occ) @ aoms_no[0])
pop1_no = np.trace(np.diag(occ) @ aoms_no[1])
print(f"Populations: {pop0_no:.4f}, {pop1_no:.4f}, Total: {pop0_no+pop1_no:.4f}")

print("\nMulliken with HF basis (but NO occupations):")
aoms_hf = [c_hf.T @ S @ eta[i] @ c_hf for i in range(mol.natm)]
pop0_hf = np.trace(np.diag(occ) @ aoms_hf[0])
pop1_hf = np.trace(np.diag(occ) @ aoms_hf[1])
print(f"Populations: {pop0_hf:.4f}, {pop1_hf:.4f}, Total: {pop0_hf+pop1_hf:.4f}")
