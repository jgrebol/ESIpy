from pyscf import gto, scf, mp, mcscf
import numpy as np
mol = gto.M(atom='H 0 0 0; F 0 0 1.1', basis='sto-3g')
mf = scf.RHF(mol).run()
mymp2 = mp.MP2(mf).run()
mycas = mcscf.CASCI(mf, 2, 2).run()

print("MP2 rdm1 shape:", mymp2.make_rdm1().shape)
print("CASCI rdm1 shape:", mycas.make_rdm1().shape)
print("Total AOs:", mol.nao)
