import os
import sys
import numpy as np

# Add current directory to path
sys.path.append(os.getcwd())

from esipy.readfchk import readfchk
from esipy import ESI
from pyscf.lo import nao

def test_nao_refactor(path):
    print(f"\n--- Testing NAO Refactor on: {path} ---")
    mol, mf = readfchk(path)
    
    # 1. Check current behavior (which uses NOs in make_aoms.py)
    # ESI internally calls make_aoms
    esi_current = ESI(mol=mol, mf=mf, partition='nao')
    aoms = esi_current.aom
    # If it's NO, aoms = [matrices, occ]
    if isinstance(aoms, list) and len(aoms) == 2 and isinstance(aoms[1], np.ndarray):
        pop_current = np.trace(aoms[1] @ sum(aoms[0]))
    else:
        pop_current = sum(2 * np.trace(m) for m in aoms)
    print(f"Current Population Sum (NAO): {pop_current:.6f}")

    # 2. Try raw NAO on correlated density
    s = mol.intor_symmetric('int1e_ovlp')
    # PySCF's nao module uses mf.make_rdm1()
    # If we pass our mf object, it uses the overwritten lambda
    try:
        u_inv = nao.nao(mol, mf, s)
        u = np.linalg.inv(u_inv)
        
        # Build AOMs manually from full correlated density
        dm = mf.make_rdm1()
        from esipy.tools import build_eta
        eta = build_eta(mol)
        
        # Trace(Dm @ S @ Eta_i @ S) 
        # or simplified in AO basis: P = Tr(Dm @ S @ U' @ Eta_ref @ U @ S) ? No.
        # Population = Tr(D @ P_i) where P_i = S @ AOM_i
        # Let's just use the Orthogonal basis U
        # D_orth = U @ S @ D @ S @ U.T
        d_orth = u_inv.T @ s @ dm @ s @ u_inv
        pop_new = np.trace(d_orth)
        print(f"Refactored Population Sum (NAO): {pop_new:.6f}")
    except Exception as e:
        print(f"Refactor failed: {e}")

test_nao_refactor('../tests/FCHK/GAUSSIAN/1_benzene_spherical.fchk')
test_nao_refactor('../tests/FCHK/GAUSSIAN/9_ccsd.fchk')
