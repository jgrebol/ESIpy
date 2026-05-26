import numpy as np
from pyscf import gto, scf
from esipy import ESI

mol = gto.M(atom='C 0 0 0; C 0 1.4 0; C 1.212 2.1 0; C 2.424 1.4 0; C 2.424 0 0; C 1.212 -0.7 0; H -0.933 0 0; H -0.933 1.4 0; H 1.212 3.033 0; H 3.357 1.4 0; H 3.357 0 0; H 1.212 -1.633 0', basis='sto-3g', spin=0)
mf = scf.UHF(mol)
mf.kernel()

esi = ESI(mol=mol, mf=mf, partition='mulliken', rings=[1,2,3,4,5,6])
print("PDI:", esi.indicators[0].pdi)
print("PDI_alpha:", esi.indicators[0].pdi_alpha)
print("PDI_beta:", esi.indicators[0].pdi_beta)
print("PDI_list:", esi.indicators[0].pdi_list)
print("AV1245:", esi.indicators[0].av1245)
