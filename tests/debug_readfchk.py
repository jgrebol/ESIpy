import sys
sys.path.insert(0, '.')
from esipy.readfchk import readfchk
mol, mf = readfchk('tests/FCHK/GAUSSIAN/11_ump2.fchk')
print("mo_occ:", mf.mo_occ)
