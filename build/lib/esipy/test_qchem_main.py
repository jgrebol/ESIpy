
import sys
import os
import numpy as np
from esipy.readfchk import Mole2, MeanField2

fchk_path = "FCHK/QCHEM/h2o_sto3g.fchk"
try:
    mol = Mole2(fchk_path)
    mf = MeanField2(fchk_path, mol)
    print(f"Is Q-Chem: {mol.fchk.is_qchem}")
    if mol.fchk.is_qchem:
        S = mf._scf.get_ovlp()
        print(f"Overlap matrix shape: {S.shape}")
        rdm1 = mf.make_rdm1()
        pop = np.trace(rdm1 @ S)
        print(f"Total population: {pop}")
        print("Q-Chem FCHK support: SUCCESS")
except Exception as e:
    print(f"Q-Chem FCHK support: FAILED with error: {e}")
