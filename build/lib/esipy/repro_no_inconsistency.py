import numpy as np
from pyscf import gto, scf
from esipy import ESI
import os
import sys

# Ensure we can import esipy
sys.path.append(os.getcwd())

mol = gto.M(atom='C 0 0 0; C 0 1.4 0; C 1.212 2.1 0; C 2.424 1.4 0; C 2.424 0 0; C 1.212 -0.7 0; H -0.933 0 0; H -0.933 1.4 0; H 1.212 3.033 0; H 3.357 1.4 0; H 3.357 0 0; H 1.212 -1.633 0', basis='sto-3g', spin=0)
mf = scf.RHF(mol).run()

from esipy.make_aoms import make_aoms
from esipy.tools import get_natorbs

# 1. Restricted calculation
esi_r = ESI(mol=mol, mf=mf, partition='mulliken', rings=[1,2,3,4,5,6])
print("\n--- RESTRICTED RESULTS ---")
esi_r.print()

# 2. Natural Orbitals calculation (using RHF density)
aoms = make_aoms(mol, mf, partition='mulliken')
S = mf.get_ovlp()
occ, coeff = get_natorbs(mf, S)

# Mock aoms to match occ dimension (all 36 AOs)
# Wait, make_aoms with partition='mulliken' for Restricted returns (Nocc, Nocc)
# We need to rebuild them for NO if we want to avoid dimension mismatch
# Or just use ESI to build them
esi_no = ESI(mol=mol, mf=mf, partition='mulliken', rings=[[1,2,3,4,5,6]])
# We can't easily force it to NO type without passing [aom, occ]
# Let's use a small hack: pass [aoms, occ] but ensure aoms are (36,36)
# Actually, make_aoms(mol, mf, 'mulliken') for RHF returns (Nocc, Nocc).
# If we want (Nao, Nao), we need to pass all coeffs.

def make_aoms_full(mol, mf):
    coeff = mf.mo_coeff # All 36 orbitals
    S = mf.get_ovlp()
    from esipy.tools import build_eta
    eta = build_eta(mol)
    aoms = []
    for i in range(mol.natm):
        aoms.append(coeff.T @ S @ eta[i] @ coeff)
    return aoms

aoms_full = make_aoms_full(mol, mf)
esi_no = ESI(aom=[aoms_full, occ], rings=[[1,2,3,4,5,6]], mol=mol, mf=mf, partition='mulliken', molinfo={"calctype": "RHF-NO", "symbols": [mol.atom_symbol(i) for i in range(mol.natm)], "basisset": "sto-3g", "energy": mf.e_tot, "method": "pyscf", "partition": "mulliken"})

print("\n--- NO RESULTS ---")
esi_no.print()
