
import sys
import os
import numpy as np
from esipy.readfchk import readfchk
from esipy import ESI

fchk_path = "FCHK/GAUSSIAN/lih_cas.fchk"
if os.path.exists(fchk_path):
    mol, mf = readfchk(fchk_path)
    # This should find "Total CI Rho(1) Density" and build NOs
    # So is_natorb will be True
    esi = ESI(mol=mol, mf=mf, partition='iao')
    print(f"ESI Tr(aom sum): {np.sum([np.trace(m) for m in esi.aom[0]])}")
    print(f"ESI Tr(occ): {np.sum(esi.aom[1])}")
    print("CASSCF FCHK test: SUCCESS")
else:
    print("CASSCF FCHK test: SKIPPED (file not found)")
