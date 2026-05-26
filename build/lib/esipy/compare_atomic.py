import os
import sys
import numpy as np
import traceback

# Add current directory to path so esipy can be imported
sys.path.append(os.getcwd())

from esipy.readfchk import readfchk
from esipy import ESI
from esipy.tools import wf_type

def get_atomic_pops(path):
    mol, mf = readfchk(path)
    esi = ESI(mol=mol, mf=mf, partition='mulliken')
    
    aoms = esi.aom
    wf = wf_type(aoms)
    
    pops = []
    if wf == "rest":
        pops = [2 * np.trace(m) for m in aoms]
    elif wf == "unrest":
        pops = [np.trace(a) + np.trace(b) for a, b in zip(aoms[0], aoms[1])]
    elif wf == "no":
        aoms_list, occ_matrix = aoms
        pops = [np.trace(occ_matrix @ m) for m in aoms_list]
    
    return np.array(pops)

def main():
    # Let's compare a few key systems
    systems = [
        '1_benzene_spherical.fchk',
        '2_benzene_cartesian.fchk',
        '9_ccsd.fchk',
        '11_ump2.fchk'
    ]
    
    base_dir = '../tests/FCHK'
    
    for sys_file in systems:
        path_g = os.path.join(base_dir, 'GAUSSIAN', sys_file)
        path_q = os.path.join(base_dir, 'QCHEM', sys_file)
        
        if not (os.path.exists(path_g) and os.path.exists(path_q)):
            print(f"Skipping {sys_file}, one of the files is missing")
            continue
            
        print(f"\n=== Comparing Atomic Populations: {sys_file} ===")
        try:
            pops_g = get_atomic_pops(path_g)
            pops_q = get_atomic_pops(path_q)
            
            diff = np.abs(pops_g - pops_q)
            max_diff = np.max(diff)
            
            print(f"Max Atomic Pop Diff: {max_diff:.6f}")
            if max_diff > 1e-4:
                print("MISMATCH detected in atomic distribution!")
                for i, (pg, pq) in enumerate(zip(pops_g, pops_q)):
                    print(f"  Atom {i+1}: G={pg:10.6f} Q={pq:10.6f} Diff={abs(pg-pq):10.6f}")
            else:
                print("Atomic distributions MATCH.")
        except Exception as e:
            print(f"Error processing {sys_file}: {e}")
            # traceback.print_exc()

if __name__ == '__main__':
    main()
