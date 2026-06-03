from tests.test5_casscf_indicators import ESItest, mol, mf, myhf, ring
import numpy as np
from esipy import ESI

esitest = ESI(mol=mol, mf=mf, myhf=myhf, rings=ring, partition='iao')
aom_data, occ = esitest.aom
inds = esitest.indicators[0]
occ_half = np.sqrt(occ)

pop_atm1 = np.einsum('i,ii->', occ, aom_data[0])
di12 = 2 * np.einsum('i,ij,j,ji->', occ_half, aom_data[0], occ_half, aom_data[1])

print(f"pop_atm1: {pop_atm1}")
print(f"di12: {di12}")
print(f"iring: {inds.iring}")
print(f"mci: {inds.mci}")
print(f"pdi: {inds.pdi}")
