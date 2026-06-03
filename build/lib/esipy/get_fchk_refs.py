import os
import numpy as np
from esipy.readfchk import readfchk
from esipy import ESI
from esipy.tools import wf_type

def get_refs(path):
    try:
        mol, mf = readfchk(path)
        res = {}
        for p in ['mulliken', 'lowdin', 'nao', 'iao']:
            try:
                esi = ESI(mol=mol, mf=mf, partition=p)
                aoms = esi.aom
                wf = wf_type(aoms)
                
                # Manual fix for misidentification in get_refs
                if wf == "no" and np.asarray(aoms[1]).ndim == 2:
                    wf = "rest"
                
                if wf == "no":
                    aom_list, occ = aoms
                    pops = np.array([np.einsum('i,ii->', occ, m) for m in aom_list])
                    from esipy.tools import find_di_no
                    di12 = find_di_no(aoms, 1, 2)
                elif wf == "unrest":
                    pops = np.array([np.trace(a) + np.trace(b) for a, b in zip(aoms[0], aoms[1])])
                    from esipy.tools import find_di
                    di12 = 2 * (find_di(aoms[0], 1, 2) + find_di(aoms[1], 1, 2))
                else:
                    pops = np.array([2 * np.trace(m) for m in aoms])
                    from esipy.tools import find_di
                    di12 = find_di(aoms, 1, 2)
                
                res[p] = {
                    'nelec': float(np.sum(pops)),
                    'pop_at1': float(pops[0]),
                    'di12': float(di12)
                }
            except Exception as e:
                # print(f"Error in {p}: {e}")
                pass
        return res
    except:
        return None

base_dir = "../tests/FCHK/GAUSSIAN"
files = [
    '1_benzene_spherical.fchk',
    '3_o2_triplet.fchk',
    '4_h2_oss.fchk',
    '7_rmp2.fchk',
    '8_cisd.fchk',
    '9_ccsd.fchk',
    '10_casscf_rest.fchk',
    '11_ump2.fchk'
]

all_refs = {}
for f in files:
    path = os.path.join(base_dir, f)
    if os.path.exists(path):
        refs = get_refs(path)
        if refs:
            all_refs[f] = refs

import json
print(json.dumps(all_refs, indent=2))
