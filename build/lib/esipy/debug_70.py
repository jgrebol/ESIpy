import numpy as np
import os
import sys
sys.path.append(os.getcwd())
from esipy.readfchk import readfchk
from esipy import ESI

def debug_70(path):
    print(f"\n--- Debugging Population 70.0 on: {path} ---")
    mol, mf = readfchk(path)
    
    print(f"  Theory: {mf.__name__}")
    print(f"  Spin: {mol.spin}")
    print(f"  Nelec: {mol.nelectron}")
    
    # Check mo_occ
    if isinstance(mf.mo_occ, list):
        print(f"  Sum(mo_occ): {np.sum(mf.mo_occ[0])} + {np.sum(mf.mo_occ[1])} = {np.sum(mf.mo_occ[0]) + np.sum(mf.mo_occ[1])}")
    else:
        print(f"  Sum(mo_occ): {np.sum(mf.mo_occ)}")

    # Check ESI Mulliken
    esi = ESI(mol=mol, mf=mf, partition='mulliken')
    from esipy.tools import wf_type
    aom = esi.aom
    wf = wf_type(aom)
    print(f"  WF Type: {wf}")
    
    if wf == "rest":
        pops = [2 * np.trace(m) for m in aom]
    elif wf == "unrest":
        pops = [np.trace(a) + np.trace(b) for a, b in zip(aom[0], aom[1])]
    elif wf == "no":
        aoms_list, occ = aom
        if occ.ndim == 1:
            pops = [occ[i] * np.trace(aoms_list[i]) for i in range(len(aoms_list))]
        else:
            pops = [np.trace(occ @ m) for m in aoms_list]
    
    print(f"  ESI Pop Sum: {np.sum(pops)}")

debug_70('../tests/FCHK/GAUSSIAN/1_benzene_spherical.fchk')
debug_70('../tests/FCHK/QCHEM/1_benzene_spherical.fchk')
