
import sys
import os
import numpy as np
from esipy.readfchk import readfchk
from esipy.make_aoms import make_aoms
from esipy.mci import compute_mci

def run_test(fchk_path):
    print(f"Testing {fchk_path}")
    mol, mf = readfchk(fchk_path)
    
    # Test IAO-EFFAO-NAO
    print("Running IAO-EFFAO-NAO...")
    aoms_iao = make_aoms(mol, mf, "iao-effao-nao")
    mci_iao = compute_mci([0,1,2], aoms_iao)
    print(f"MCI IAO (first few): {mci_iao}")
    
    # Test PEIAO
    print("Running PEIAO...")
    aoms_peiao = make_aoms(mol, mf, "peiao")
    mci_peiao = compute_mci([0,1,2], aoms_peiao)
    print(f"MCI PEIAO (first few): {mci_peiao}")
    
    return mci_iao, mci_peiao

if __name__ == "__main__":
    fchk = "../FCHK/GAUSSIAN/h2o.fchk"
    if not os.path.exists(fchk):
        print(f"FCHK file {fchk} not found")
        sys.exit(1)
    
    m1, m2 = run_test(fchk)
    np.save("baseline_iao.npy", m1)
    np.save("baseline_peiao.npy", m2)
    print("Baseline saved.")
