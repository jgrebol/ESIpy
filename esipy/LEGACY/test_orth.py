import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))
from esipy.readfchk import readfchk
import numpy as np

path = '../joan/LiH/GS/lih_1.0.fchk'
mol, mf, myhf = readfchk(path)
S = mol.intor('int1e_ovlp')
C = mf.mo_coeff
orth = C.T @ S @ C
diag = np.diag(orth)
print("Diagonal of C.T @ S @ C:", diag[:10])
print("Max deviation from 1.0:", np.max(np.abs(diag - 1.0)))
print("Min/Max of full orth matrix:", np.min(orth), np.max(orth))
print("Sum of occupations:", np.sum(mf.mo_occ))
