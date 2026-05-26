import numpy as np
from pyscf import gto, scf
from esipy.tools import wf_type
from esipy.indicators import compute_pdi, compute_iring
from esipy import mci
from esipy.make_aoms import make_aoms

mol = gto.M(atom='C 0 0 0; C 0 1.4 0; C 1.212 2.1 0; C 2.424 1.4 0; C 2.424 0 0; C 1.212 -0.7 0; H -0.933 0 0; H -0.933 1.4 0; H 1.212 3.033 0; H 3.357 1.4 0; H 3.357 0 0; H 1.212 -1.633 0', basis='sto-3g')
mf = scf.RHF(mol).run()
aoms = make_aoms(mol, mf, 'mulliken')

# Restricted:
print("Restricted DI (aom[0] @ aom[3]):", 4 * np.trace(aoms[0] @ aoms[3]))
print("compute_pdi with restricted aom:", compute_pdi([1,2,3,4,5,6], aoms))

mf_u = scf.UHF(mol).run()
aoms_u = make_aoms(mol, mf_u, 'mulliken')
print("Unrestricted DI alpha (aom[0] @ aom[3]):", 2 * np.trace(aoms_u[0][0] @ aoms_u[0][3]))
print("compute_pdi with unrestricted aom alpha:", compute_pdi([1,2,3,4,5,6], aoms_u[0]))
