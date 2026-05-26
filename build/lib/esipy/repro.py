import sys
sys.path.append('.')
from esipy.readfchk import readfchk
from esipy import ESI

mol, mf = readfchk('../tests/FCHK/GAUSSIAN/unrest.fchk')
try:
    esi = ESI(mol=mol, mf=mf, partition='iao', minlen=3, rings=[[1,2,3]])
except Exception as e:
    import traceback
    traceback.print_exc()
