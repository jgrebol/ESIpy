from pyscf import gto, scf, mp, mcscf
import numpy as np
mol = gto.M(atom='H 0 0 0; F 0 0 1.1', basis='sto-3g')
mf = scf.RHF(mol).run()
mycas = mcscf.CASCI(mf, 2, 2).run()

S = mol.intor_symmetric('int1e_ovlp')
dm1_cas = mycas.make_rdm1()
print("Trace(dm1):", np.trace(dm1_cas))
print("Trace(dm1 @ S):", np.trace(dm1_cas @ S))
print("Electrons:", mol.nelectron)
