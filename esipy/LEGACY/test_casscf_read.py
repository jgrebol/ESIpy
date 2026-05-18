
import sys
import os
import numpy as np
from esipy.readfchk import readfchk
from esipy.make_aoms import make_aoms

def test_casscf(fchk_path):
    print(f"--- Testing CASSCF Read: {fchk_path} ---")
    # 1. Test reading
    mol, mf = readfchk(fchk_path)
    
    # Check if mf has mo_occ with fractional values
    occ = np.asarray(mf.mo_occ)
    print(f"Found {len(occ)} orbitals.")
    print(f"Top 10 occupations: {occ[:10]}")
    
    fractional = np.any((occ > 1e-6) & (np.abs(occ - 1.0) > 1e-6) & (np.abs(occ - 2.0) > 1e-6))
    if fractional:
        print("SUCCESS: Fractional occupations (Natural Orbitals) detected.")
    else:
        print("FAILURE: No fractional occupations detected (expected for CASSCF).")

    # 2. Test AOM construction with DPEIAO
    print("\n--- Testing DPEIAO Construction ---")
    try:
        # DPEIAO(0.5) blends IAO-EFFAO and PEIAO
        aom_dpeiao = make_aoms(mol, mf, "dpeiao(0.5)")
        print("SUCCESS: DPEIAO construction completed.")
        
        # AOMs for Natural Orbitals should be a list [aoms, occ_matrix]
        if isinstance(aom_dpeiao, list) and len(aom_dpeiao) == 2 and isinstance(aom_dpeiao[1], np.ndarray):
            print("SUCCESS: AOM return format for Natural Orbitals is correct [aoms, occ_matrix].")
        else:
            print(f"DEBUG: AOM return type: {type(aom_dpeiao)}")
            if isinstance(aom_dpeiao, list):
                 print(f"DEBUG: List length: {len(aom_dpeiao)}")
    except Exception as e:
        print(f"FAILURE: DPEIAO construction failed with error: {e}")

if __name__ == "__main__":
    fchk = "/home/joan/PycharmProjects/ESIpy/joan/LiH/GS/lih_1.5.fchk"
    if not os.path.exists(fchk):
        print(f"FCHK file {fchk} not found")
        sys.exit(1)
    
    test_casscf(fchk)
