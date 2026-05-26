
import os
import numpy as np
from esipy.readfchk import readfchk
from esipy.make_aoms import make_aoms

bz_fchk = '../tests/FCHK/GAUSSIAN/bz.fchk'
if not os.path.exists(bz_fchk):
    print("bz.fchk not found")
    exit()

mol, mf = readfchk(bz_fchk)
print(f"Mole NAO: {mol.nao_nr()}")
print(f"MF Overlap shape: {mf.get_ovlp().shape}")
print(f"MF MO Coeff shape: {np.shape(mf.mo_coeff)}")

try:
    aoms = make_aoms(mol, mf, 'iao-effao-nao')
    print("Success!")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
