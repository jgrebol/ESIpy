import sys
import os
# Add the parent directory of 'esipy' to sys.path
sys.path.append(os.path.dirname(os.getcwd()))
from esipy.readfchk import readfchk
import numpy as np

path = '../joan/OUISSAM/lih.fchk'
mol, mf, myhf = readfchk(path)

print("Primary mf type:", type(mf))
print("Primary mf name:", getattr(mf, '__name__', 'None'))
if hasattr(mf, 'mo_occ'):
    print("Primary mf occupations:", mf.mo_occ)
    is_no = not np.all((np.abs(mf.mo_occ - 2.0) < 1e-4) | (np.abs(mf.mo_occ - 0.0) < 1e-4))
    print("Is NO:", is_no)

if myhf:
    print("myhf type:", type(myhf))
    print("myhf name:", getattr(myhf, '__name__', 'None'))
    if hasattr(myhf, 'mo_occ'):
        print("myhf occupations:", myhf.mo_occ)
else:
    print("myhf is None")
