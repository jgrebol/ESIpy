import os
import numpy as np
import itertools
from esipy.readfchk import readfchk
from esipy import ESI
from numpy import trace, dot
from esipy.readfchk import Mole2, MeanField2
import esipy.readfchk as readfchk_mod

def get_pop_li(esi):
    pops = [2.0 * np.trace(m) for m in esi.aom]
    lis = [2.0 * np.einsum('ij,ji->', m, m) for m in esi.aom]
    return np.array(pops), np.array(lis)

fchk_dir = "../FCHK"
g_path = os.path.join(fchk_dir, "GAUSSIAN", "bz.fchk")
q_path = os.path.join(fchk_dir, "QCHEM", "bz.fchk")

mol_g, mf_g = readfchk(g_path)
esi_g = ESI(mol=mol_g, mf=mf_g, partition='mulliken')
g_pops, g_lis = get_pop_li(esi_g)

def read_q_raw(path):
    mol2 = Mole2(path)
    mol2.build()
    mf2 = MeanField2(path, mol2)
    mo_flat = readfchk_mod.read_list_from_fchk('Alpha MO coefficients', path)
    mo_arr = np.array(mo_flat, dtype=float).reshape(mf2.nummo, mf2.nao).T
    return mol2, mo_arr, mf2

mol2_q, mo_raw_q, mf2_q = read_q_raw(q_path)

coord_perms = list(itertools.permutations(range(3)))
print(f"Testing {len(coord_perms)} coordinate permutations...")

best_err = 1e10
best_c = None

for cp in coord_perms:
    # Q-Chem coordinates: mol2_q.coord
    # Gaussian coordinates: mol_g.atom_coords()
    # We need to permute Q-Chem coordinates and see if they match Gaussian
    q_coords_perm = mol2_q.coord[:, cp]
    err = np.linalg.norm(mol_g.atom_coords() - q_coords_perm)
    if err < best_err:
        best_err = err
        best_c = cp
        print(f"New best coord err: {best_err:.6f} | Coord Perm: {cp}")

print(f"\nFinal Best Coord Perm: {best_c} | Error: {best_err}")

# Now let's try to match populations with permuted atoms
atom_perms = list(itertools.permutations(range(mol2_q.natm)))
# Too many for 12 atoms (12! is huge). Let's assume atoms are in same order but coords permuted.
# Actually, let's see if the coordinates match after permutation.
