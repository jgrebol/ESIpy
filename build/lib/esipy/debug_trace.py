import numpy as np
from esipy.readfchk import readfchk
from pyscf import gto

def debug_file(path):
    print(f"\n--- Debugging: {path} ---")
    mol_f, mf_f = readfchk(path)
    s = mf_f.get_ovlp()
    c = mf_f.mo_coeff
    
    if isinstance(c, list):
        print("UHF case")
        for i, ci in enumerate(c):
            # Check orthogonality
            ovlp_mo = ci.T @ s @ ci
            diag = np.diag(ovlp_mo)
            print(f"  Channel {i} diag min/max: {np.min(diag):.4f} / {np.max(diag):.4f}")
            print(f"  Channel {i} off-diag max: {np.max(np.abs(ovlp_mo - np.diag(diag))):.4f}")
            # Check population of first few atoms
            dm = np.dot(ci * mf_f.mo_occ[i], ci.T)
            nelec = np.trace(dm @ s)
            print(f"  Channel {i} trace(PS): {nelec:.4f}")
    else:
        print("RHF case")
        ovlp_mo = c.T @ s @ c
        diag = np.diag(ovlp_mo)
        print(f"  Diag min/max: {np.min(diag):.4f} / {np.max(diag):.4f}")
        print(f"  Off-diag max: {np.max(np.abs(ovlp_mo - np.diag(diag))):.4f}")
        dm = mf_f.make_rdm1()
        nelec = np.trace(dm @ s)
        print(f"  Trace(PS): {nelec:.4f}")

debug_file("../tests/FCHK/GAUSSIAN/3_o2_triplet.fchk")
debug_file("../tests/FCHK/GAUSSIAN/2_benzene_cartesian.fchk")
