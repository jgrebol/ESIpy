import numpy as np
from pyscf import gto, scf, lo
from esipy.readfchk import readfchk
from esipy import ESI
from esipy.tools import wf_type
import os

def check_nao_iao(prog, filename):
    path = f"../tests/FCHK/{prog}/{filename}"
    if not os.path.exists(path): return
    
    print(f"\n=== NAO/IAO Check: {prog} {filename} ===")
    mol_f, mf_f = readfchk(path)
    
    for p in ['nao', 'iao']:
        try:
            esi = ESI(mol=mol_f, mf=mf_f, partition=p)
            aoms_data = esi.aom
            wf = wf_type(aoms_data)
            
            if wf == "no":
                aoms, occ = aoms_data
                pops = np.array([np.einsum('i,ii->', occ, m) for m in aoms])
            elif wf == "unrest":
                a_aoms, b_aoms = aoms_data
                pops = np.array([np.trace(a) + np.trace(b) for a, b in zip(a_aoms, b_aoms)])
            else:
                pops = np.array([2 * np.trace(m) for m in aoms_data])
                
            pop_sum = np.sum(pops)
            print(f"  Partition {p:6}: Sum(Pops) = {pop_sum:.6f}")
            # Benzene is 42, Water is 10
            if abs(pop_sum - 42.0) < 0.1 or abs(pop_sum - 10.0) < 0.1:
                print(f"  Partition {p:6}: SUCCESS (Electron count preserved)")
            else:
                print(f"  Partition {p:6}: FAILURE (Sum(Pops) = {pop_sum:.6f})")
        except Exception as e:
            print(f"  Partition {p:6}: FAILED: {e}")

check_nao_iao('GAUSSIAN', '1_benzene_spherical.fchk')
check_nao_iao('QCHEM', '7_rmp2.fchk')
