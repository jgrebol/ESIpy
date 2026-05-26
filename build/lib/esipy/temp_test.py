
import sys
import os
sys.path.insert(0, '/home/joan/PycharmProjects/ESIpy')
import numpy as np
from esipy.readfchk import Mole2, MeanField2
try:
    mol = Mole2('/home/joan/PycharmProjects/ESIpy/FCHK/QCHEM/h2o_sto3g.fchk')
    mf = MeanField2('/home/joan/PycharmProjects/ESIpy/FCHK/QCHEM/h2o_sto3g.fchk', mol)
    print(f"Branch: ESIpy")
    print(f"Label: Q-Chem")
    print(f"Calculated type: {mf.__name__}")
    if hasattr(mf, 'mo_occ'):
        if isinstance(mf.mo_occ, list):
            print(f"Alpha occ shape: {np.shape(mf.mo_occ[0])}")
            print(f"Beta occ shape: {np.shape(mf.mo_occ[1])}")
        else:
            print(f"Occ shape: {np.shape(mf.mo_occ)}")
    
    # Natural Orbitals check
    if hasattr(mf, 'mo_occ') and not isinstance(mf.mo_occ, list):
        occ = mf.mo_occ
        is_natorb = np.any((occ > 1e-6) & (np.abs(occ - 1.0) > 1e-6) & (np.abs(occ - 2.0) > 1e-6))
        print(f"Is Natural Orbital: {is_natorb}")
    
    # Q-Chem check
    if hasattr(mol.fchk, 'is_qchem'):
        print(f"Is Q-Chem: {mol.fchk.is_qchem}")
        if mol.fchk.is_qchem:
            S = mf._scf.get_ovlp()
            print(f"Overlap matrix shape: {S.shape}")
            rdm1 = mf.make_rdm1()
            if isinstance(rdm1, np.ndarray) and rdm1.ndim == 3:
                 pop = np.trace(rdm1[0] @ S) + np.trace(rdm1[1] @ S)
            else:
                 pop = np.trace(rdm1 @ S)
            print(f"Total population: {pop}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
