
import time
from esipy.readfchk import MeanField2, Mole2
import os

fchk_path = '../tests/FCHK/GAUSSIAN/h2o_sto3g.fchk'
if not os.path.exists(fchk_path):
    fchk_path = 'tests/FCHK/GAUSSIAN/lih_cas.fchk'

print(f"Profiling reading {fchk_path}...")
start = time.time()
mol2 = Mole2(fchk_path)
mf2 = MeanField2(fchk_path, mol2)
end = time.time()
print(f"Time taken: {end - start:.4f} seconds")
