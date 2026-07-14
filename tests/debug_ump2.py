import sys
sys.path.insert(0, '.')
from esipy.readfchk import readfchk
from esipy.make_aoms import make_aoms
from esipy.tools import get_natorbs, wf_type, is_correlated
mol, mf = readfchk('tests/FCHK/GAUSSIAN/11_ump2.fchk')
S = mf.get_ovlp()
occ, coeff = get_natorbs(mf, S)
print("Occupancies:", occ)
print("is_correlated:", is_correlated(occ))
from esipy import ESI
esi = ESI(mol, mf)
print("aoms len:", len(esi.aom))
print("wf_type:", wf_type(esi.aom))
