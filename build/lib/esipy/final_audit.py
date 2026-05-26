import numpy as np
import os
import sys
sys.path.append(os.getcwd())
from esipy.readfchk import readfchk

def audit_system(filename):
    print(f"\n# System: {filename}")
    for src in ['GAUSSIAN', 'QCHEM']:
        path = os.path.join('../tests/FCHK', src, filename)
        if not os.path.exists(path):
            print(f"  [{src:<8}] MISSING FILE")
            continue
            
        try:
            mol, mf = readfchk(path)
            # Find which label was used for density
            with open(path, 'r') as f:
                content = f.read()
            
            density_label = "None"
            for lbl in ['Total CI Rho(1) Density', 'Total CI Density', 'Total CC Density', 'Total MP2 Density', 'Total SCF Density']:
                if lbl in content:
                    density_label = lbl
                    break
            
            # Check occupations to see if it's really correlated
            occ = mf.mo_occ
            if isinstance(occ, list):
                top3 = occ[0][:3]
                is_corr = any(abs(o - 1.0) > 1e-4 for o in top3) # For UHF, occupied is 1.0
                actual_pop = np.sum(occ[0]) + np.sum(occ[1])
            else:
                top3 = occ[:3]
                is_corr = any(abs(o - 2.0) > 1e-4 for o in top3) # For RHF, occupied is 2.0
                actual_pop = np.sum(occ)
            
            status = "PASS" if abs(actual_pop - mol.nelectron) < 0.1 else "POP_MISMATCH"
            
            print(f"  [{src:<8}] {status:<12} | Label: {density_label:<25} | Correlated: {str(is_corr):<5} | Pop: {actual_pop:.2f}/{mol.nelectron}")
            if is_corr:
                print(f"             Top Occ: {top3}")
                
        except Exception as e:
            print(f"  [{src:<8}] ERROR: {str(e)}")

filenames = [
    '1_benzene_spherical.fchk', '2_benzene_cartesian.fchk', '3_o2_triplet.fchk',
    '4_h2_oss.fchk', '5_high_l.fchk', '6_ecp.fchk', '7_rmp2.fchk',
    '8_cisd.fchk', '9_ccsd.fchk', '10_casscf_rest.fchk', '11_ump2.fchk',
    '12_casscf_unrest.fchk'
]

for f in filenames:
    audit_system(f)
