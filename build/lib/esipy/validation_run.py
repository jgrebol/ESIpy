import os
import sys
import numpy as np
import traceback

# Add current directory to path so esipy can be imported
sys.path.append(os.getcwd())

from esipy.readfchk import readfchk
from esipy import ESI
from esipy.tools import wf_type

def test_fchk(path):
    try:
        mol, mf = readfchk(path)
        # Check trace sum (population)
        esi = ESI(mol=mol, mf=mf, partition='mulliken')
        
        # In esipy, aom trace sum should be number of electrons
        nelec_read = mol.nelectron
        pop_sum = 0
        
        aoms = esi.aom
        wf = wf_type(aoms)
        
        if wf == "rest":
            # aoms is a list of matrices
            pop_sum = sum(2 * np.trace(m) for m in aoms)
        elif wf == "unrest":
            # aoms = [aoms_alpha, aoms_beta]
            pop_sum = sum(np.trace(m) for m in aoms[0]) + sum(np.trace(m) for m in aoms[1])
        elif wf == "no":
            # aoms = [aoms_list, occupations]
            aoms_list, occ_matrix = aoms
            pop_sum = sum(np.trace(occ_matrix @ m) for m in aoms_list)
        
        if isinstance(pop_sum, np.ndarray):
            pop_sum = pop_sum.item()
            
        return True, float(pop_sum), int(nelec_read)
    except Exception as e:
        return False, str(e), traceback.format_exc()

def main():
    base_dir = '../tests/FCHK'
    folders = ['GAUSSIAN', 'QCHEM']
    
    results = {}
    
    for folder in folders:
        folder_path = os.path.join(base_dir, folder)
        if not os.path.exists(folder_path):
            print(f"Folder {folder_path} not found")
            continue
            
        print(f"\n--- Testing {folder} ---")
        files = [f for f in os.listdir(folder_path) if f.endswith('.fchk')]
        files.sort()
        
        for f in files:
            path = os.path.join(folder_path, f)
            success, val1, val2 = test_fchk(path)
            if success:
                print(f"[OK] {f:30} Pop: {val1:10.4f} / {val2:2.0f}")
                results[(folder, f)] = (val1, val2)
            else:
                print(f"[FAIL] {f:30} Error: {val1}")
                results[(folder, f)] = None

    # Comparison
    print("\n--- Cross-Program Comparison ---")
    common_files = set(f for (fld, f) in results if fld == 'GAUSSIAN') & \
                   set(f for (fld, f) in results if fld == 'QCHEM')
    
    for f in sorted(list(common_files)):
        res_g = results[('GAUSSIAN', f)]
        res_q = results[('QCHEM', f)]
        
        if res_g and res_q:
            diff = abs(res_g[0] - res_q[0])
            status = "MATCH" if diff < 1e-4 else "MISMATCH"
            print(f"{f:30} G: {res_g[0]:.4f} Q: {res_q[0]:.4f} Diff: {diff:.4f} {status}")

if __name__ == '__main__':
    main()
