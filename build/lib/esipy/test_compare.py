import numpy as np
from pyscf import gto, scf
from esipy import ESI
mol = gto.M(atom='C 0 0 0; C 0 1.4 0; C 1.212 2.1 0; C 2.424 1.4 0; C 2.424 0 0; C 1.212 -0.7 0; H -0.933 0 0; H -0.933 1.4 0; H 1.212 3.033 0; H 3.357 1.4 0; H 3.357 0 0; H 1.212 -1.633 0', basis='sto-3g', spin=0)
mf_r = scf.RHF(mol).run()
mf_u = scf.UHF(mol).run()

esi_r = ESI(mol=mol, mf=mf_r, partition='mulliken', rings=[1,2,3,4,5,6])
esi_u = ESI(mol=mol, mf=mf_u, partition='mulliken', rings=[1,2,3,4,5,6])

print("RHF PDI:", esi_r.indicators[0].pdi)
print("UHF PDI:", esi_u.indicators[0].pdi)
print("RHF AV1245:", esi_r.indicators[0].av1245)
print("UHF AV1245:", esi_u.indicators[0].av1245)
