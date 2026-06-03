import os
import numpy as np
import pickle
from esipy.readfchk import readfchk
from esipy import ESI
from esipy.tools import wf_type

def analyze_file(path):
    mol_f, mf_f = readfchk(path)
    s = mol_f.intor_symmetric('int1e_ovlp')
    dm = mf_f.make_rdm1()
    
    # Target Physics (What a perfect read should give)
    if dm.ndim == 3: # UHF
        nelec = np.trace((dm[0] + dm[1]) @ s)
    else: # RHF / NO
        nelec = np.trace(dm @ s)
    
    print(f"\n--- FILE: {os.path.basename(path)} ---")
    print(f"  Target Nelec: {mol_f.pyscf_mol.nelectron}")
    print(f"  Actual Nelec: {nelec:.10f}")
    
    # Orthonormality Check
    c = mf_f.mo_coeff
    if isinstance(c, list):
        ortho = max(np.max(np.abs(ci.T @ s @ ci - np.eye(ci.shape[1]))) for ci in c)
    else:
        ortho = np.max(np.abs(c.T @ s @ c - np.eye(c.shape[1])))
    print(f"  Ortho Error:  {ortho:.2e}")

files = ["GAUSSIAN/3_o2_triplet.fchk", "GAUSSIAN/4_h2_oss.fchk", "GAUSSIAN/10_casscf_rest.fchk"]
for f in files: analyze_file(os.path.join("../tests/FCHK", f))
