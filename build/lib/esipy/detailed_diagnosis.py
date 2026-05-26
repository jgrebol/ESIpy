import os
import sys
import numpy as np
# Add current directory to path
sys.path.append(os.getcwd())

from esipy.readfchk import readfchk
from esipy import ESI
from pyscf import gto, scf

def get_total_pop(mol, mf):
    from esipy.tools import wf_type
    esi = ESI(mol=mol, mf=mf, partition='mulliken')
    aoms = esi.aom
    wf = wf_type(aoms)
    if wf == "rest":
        return sum(2 * np.trace(m) for m in aoms)
    elif wf == "unrest":
        return sum(np.trace(a) + np.trace(b) for a, b in zip(aoms[0], aoms[1]))
    elif wf == "no":
        aoms_list, occ_matrix = aoms
        if occ_matrix.ndim == 1:
            return sum(occ_matrix[i] * np.trace(aoms_list[i]) for i in range(len(aoms_list)))
        else:
            return np.trace(occ_matrix @ sum(aoms_list)) # Simplified trace sum
    return 0.0

def main():
    base_dir = '../tests/FCHK'
    filenames = [
        '1_benzene_spherical.fchk', '2_benzene_cartesian.fchk', '3_o2_triplet.fchk',
        '4_h2_oss.fchk', '5_high_l.fchk', '6_ecp.fchk', '7_rmp2.fchk',
        '8_cisd.fchk', '9_ccsd.fchk', '10_casscf_rest.fchk', '11_ump2.fchk',
        '12_casscf_unrest.fchk'
    ]

    print(f"{'System':<25} | {'Source':<10} | {'Status':<10} | {'Pop Sum':<10} | {'Expected':<8}")
    print("-" * 75)

    for f in filenames:
        expected = None
        for src in ['GAUSSIAN', 'QCHEM']:
            path = os.path.join(base_dir, src, f)
            if not os.path.exists(path):
                print(f"{f:<25} | {src:<10} | MISSING    | {'-':<10} | -")
                continue
            
            try:
                mol, mf = readfchk(path)
                if expected is None: expected = mol.nelectron
                pop = get_total_pop(mol, mf)
                status = "OK" if abs(pop - expected) < 0.1 else "MISMATCH"
                print(f"{f:<25} | {src:<10} | {status:<10} | {pop:10.4f} | {expected:<8}")
            except Exception as e:
                print(f"{f:<25} | {src:<10} | ERROR      | {str(e)[:10]:<10} | -")

if __name__ == '__main__':
    main()
