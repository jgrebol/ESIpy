from pyscf import gto, scf, mcscf
import numpy as np

mol = gto.M(atom='Li 0 0 0; F 0 0 1.6', basis='aug-cc-pVDZ', spin=0, symmetry=True)
mf = scf.RHF(mol).run()
mycas = mcscf.CASSCF(mf, 6, 6)
mycas.fcisolver.nroots = 4
mycas = mycas.state_average([0.25, 0.25, 0.25, 0.25])
mycas.kernel()

np.save('lif_1.6_mo.npy', mycas.mo_coeff)
print("Saved perfect MOs.")
