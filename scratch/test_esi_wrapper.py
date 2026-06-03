from pyscf import gto, scf, mp, mcscf
from esipy import ESI
import numpy as np

mol = gto.M(atom='H 0 0 0; F 0 0 1.1', basis='sto-3g')
mf = scf.RHF(mol).run()

print("\n--- Testing Raw MP2 ---")
mymp2 = mp.MP2(mf).run()
esi_mp2 = ESI(mol=mol, mf=mymp2, partition='mulliken', rings=[[0,1]])
print("Mulliken AOM elements:", np.sum(np.diag(esi_mp2.aom[0])))

print("\n--- Testing Raw CASCI ---")
mycas = mcscf.CASCI(mf, 2, 2).run()
esi_cas = ESI(mol=mol, mf=mycas, partition='mulliken', rings=[[0,1]])
print("Mulliken AOM elements:", np.sum(np.diag(esi_cas.aom[0])))
