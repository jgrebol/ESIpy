from pyscf import gto, scf, mp, mcscf
from esipy import ESI
import numpy as np

mol = gto.M(atom='H 0 0 0; F 0 0 1.1', basis='sto-3g')
mf = scf.RHF(mol).run()
mymp2 = mp.MP2(mf).run()
esi_mp2 = ESI(mol=mol, mf=mymp2, partition='mulliken', rings=[[0,1]])
print("Type of aom:", type(esi_mp2.aom))
print("Len of aom:", len(esi_mp2.aom))
print("Shape of aom[0]:", getattr(esi_mp2.aom[0], 'shape', type(esi_mp2.aom[0])))
