import numpy as np
from pyscf import gto, scf
from esipy import ESI

mol = gto.M(atom='C 0 0 0; C 0 1.4 0; C 1.212 2.1 0; C 2.424 1.4 0; C 2.424 0 0; C 1.212 -0.7 0; H -0.933 0 0; H -0.933 1.4 0; H 1.212 3.033 0; H 3.357 1.4 0; H 3.357 0 0; H 1.212 -1.633 0', basis='sto-3g', spin=2)
mf = scf.UHF(mol)
mf.kernel()

from esipy.make_aoms import make_aoms
from esipy.indicators import compute_pdi
aoms = make_aoms(mol, mf, 'mulliken')
print("PDI_alpha compute_pdi direct:", compute_pdi([1,2,3,4,5,6], aoms[0]))
print("PDI_beta compute_pdi direct:", compute_pdi([1,2,3,4,5,6], aoms[1]))
