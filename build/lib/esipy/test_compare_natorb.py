import numpy as np
from pyscf import gto, scf
from esipy import ESI
mol = gto.M(atom='C 0 0 0; C 0 1.4 0; C 1.212 2.1 0; C 2.424 1.4 0; C 2.424 0 0; C 1.212 -0.7 0; H -0.933 0 0; H -0.933 1.4 0; H 1.212 3.033 0; H 3.357 1.4 0; H 3.357 0 0; H 1.212 -1.633 0', basis='sto-3g', spin=0)
mf_r = scf.RHF(mol).run()

# Hack natural orbitals out of it (essentially just passing the RHF aom but wrapping it as natorb)
from esipy.make_aoms import make_aoms
from esipy.tools import get_natorbs
aom = make_aoms(mol, mf_r, partition='mulliken')
S = mf_r.get_ovlp()
occ, coeff = get_natorbs(mf_r, S)

esi_no = ESI(aom=[aom, occ], rings=[[1,2,3,4,5,6]], mol=mol, mf=mf_r, partition='mulliken', molinfo={"calctype": "RHF", "symbols": ["C", "C", "C", "C", "C", "C", "H", "H", "H", "H", "H", "H"]})
print("NO PDI:", esi_no.indicators[0].pdi)
print("NO AV1245:", esi_no.indicators[0].av1245)
