import sys
sys.path.insert(0, '.')
from esipy.readfchk import readfchk
mol, mf = readfchk('tests/FCHK/GAUSSIAN/11_ump2.fchk')
print("mf type:", type(mf))
print("mo_occ:", mf.mo_occ)
rdm1 = mf.make_rdm1()
print("rdm1 type:", type(rdm1), "shape:", rdm1.shape)
from scipy.linalg import eigh
S = mf.get_ovlp()
occ, coeff = eigh(S @ rdm1 @ S, b=S)
print("direct eigh occ:", occ)
