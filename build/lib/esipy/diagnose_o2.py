import numpy as np
import os
import sys
sys.path.append(os.getcwd())
from esipy.readfchk import readfchk

def diagnose_qchem_o2(filename):
    path = os.path.join('../tests/FCHK/QCHEM', filename)
    if not os.path.exists(path): return
    
    print(f"\n=== Diagnosing {filename} ===")
    mol, mf = readfchk(path)
    
    s = mol.intor_symmetric('int1e_ovlp')
    dm = mf.make_rdm1()
    
    if dm.ndim == 3:
        da, db = dm
        pop_a = np.trace(da @ s)
        pop_b = np.trace(db @ s)
        print(f"  Alpha Pop: {pop_a:.4f}")
        print(f"  Beta Pop:  {pop_b:.4f}")
        print(f"  Total Pop: {pop_a + pop_b:.4f} (Expected: {mol.nelectron})")
        print(f"  Sum(mo_occ_a): {np.sum(mf.mo_occ[0]):.4f}")
        print(f"  Sum(mo_occ_b): {np.sum(mf.mo_occ[1]):.4f}")
    else:
        pop = np.trace(dm @ s)
        print(f"  Total Pop: {pop:.4f} (Expected: {mol.nelectron})")
        print(f"  Sum(mo_occ): {np.sum(mf.mo_occ):.4f}")

diagnose_qchem_o2('3_o2_triplet.fchk')
diagnose_qchem_o2('1_benzene_spherical.fchk')
diagnose_qchem_o2('2_benzene_cartesian.fchk')
