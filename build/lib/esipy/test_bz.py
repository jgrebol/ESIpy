import numpy as np
from esipy.readfchk import readfchk
bz_fchk = '../tests/FCHK/GAUSSIAN/bz.fchk'
mol, mf = readfchk(bz_fchk)
print("Type of mf:", type(mf))
print("mf name:", getattr(mf, "__name__", "None"))
print("mo_coeff type:", type(mf.mo_coeff))
print("mo_coeff shape:", np.shape(mf.mo_coeff))
