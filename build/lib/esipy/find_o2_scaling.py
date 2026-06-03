import numpy as np
from esipy.readfchk import readfchk, read_list_from_fchk
import os

path = "../tests/FCHK/GAUSSIAN/3_o2_triplet.fchk"
mol, mf = readfchk(path)
S_p = mol.pyscf_mol.intor_symmetric('int1e_ovlp')
s_diag = np.diag(S_p)
v = np.sqrt(s_diag)

dt_flat = read_list_from_fchk('Total SCF Density', path)
n = mol.nao
P_raw = np.zeros((n, n))
tril_idx = np.tril_indices(n)
P_raw[tril_idx] = dt_flat
P_raw = P_raw + P_raw.T - np.diag(np.diag(P_raw))

P_scaled = P_raw * v[:, None] * v[None, :]
print(f"Trace(P_scaled @ S_pyscf): {np.trace(P_scaled @ S_p):.4f}")
