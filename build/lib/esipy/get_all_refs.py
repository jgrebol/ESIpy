import numpy as np
import os
from pyscf import gto, scf, mcscf, cc, ci, mp
from esipy import ESI
from esipy.readfchk import readfchk

def get_esi_vals(mol, mf, myhf=None, partition='iao'):
    ring = [1, 2, 3, 4, 5, 6] if mol.natm >= 6 else []
    esi = ESI(mol=mol, mf=mf, myhf=myhf, rings=ring, partition=partition)
    
    aom_data = esi.aom
    if isinstance(aom_data, list) and len(aom_data) == 2 and not isinstance(aom_data[0], np.ndarray):
        aom, occ = aom_data
    else:
        aom = aom_data
        occ = np.asarray(mf.mo_occ)
        if occ.ndim == 2: occ = occ[0] # RHF
        
    pop_at1 = np.einsum('i,ii->', occ, aom[0])
    
    res = {'pop_at1': round(float(pop_at1), 6), 'nelec': round(float(np.sum(occ)), 6)}
    
    if ring:
        ind = esi.indicators[0]
        res.update({
            'iring': round(float(ind.iring), 6),
            'mci': round(float(ind.mci), 6),
            'av1245': round(float(ind.av1245), 6),
            'pdi': round(float(ind.pdi), 6)
        })
    
    # DI 1-2
    from esipy.tools import find_di_no, find_di
    if len(aom_data) == 2 and not isinstance(aom_data[0], np.ndarray):
        di12 = find_di_no(aom_data, 1, 2)
    else:
        di12 = 2 * find_di(aom, 1, 2)
    res['di12'] = round(float(di12), 6)
    
    return res

# 1. Benzene STO-3G
atom_bz = """
    C        0.000000000      0.000000000      1.393096000
    C        0.000000000      1.206457000      0.696548000
    C        0.000000000      1.206457000     -0.696548000
    C        0.000000000      0.000000000     -1.393096000
    C        0.000000000     -1.206457000     -0.696548000
    C        0.000000000     -1.206457000      0.696548000
    H        0.000000000      0.000000000      2.483127000
    H        0.000000000      2.150450000      1.241569000
    H        0.000000000      2.150450000     -1.241569000
    H        0.000000000      0.000000000     -2.483127000
    H        0.000000000     -2.150450000     -1.241569000
    H        0.000000000     -2.150450000      1.241569000
"""
mol_bz = gto.M(atom=atom_bz, basis='sto-3g')
mf_bz = scf.RHF(mol_bz).run()

print('--- TEST 5 (CASSCF IAO Benzene) ---')
mc = mcscf.CASSCF(mf_bz, 6, 6).run()
print(get_esi_vals(mol_bz, mc, mf_bz))

print('\n--- TEST 6 (CCSD IAO Benzene) ---')
mycc = cc.CCSD(mf_bz).run()
mycc.solve_lambda()
print(get_esi_vals(mol_bz, mycc, mf_bz))

# 2. Water STO-3G (for test9 cases)
atom_h2o = 'O 0 0 0; H 0 0.757 0.586; H 0 -0.757 0.586'
mol_h2o = gto.M(atom=atom_h2o, basis='sto-3g')
mf_h2o = scf.RHF(mol_h2o).run()

print('\n--- TEST 9 (MP2 IAO Water) ---')
mmp = mp.MP2(mf_h2o).run()
print(get_esi_vals(mol_h2o, mmp, mf_h2o))

print('\n--- TEST 9 (CISD IAO Water) ---')
mci = ci.CISD(mf_h2o).run()
print(get_esi_vals(mol_h2o, mci, mf_h2o))

print('\n--- TEST 9 (CCSD IAO Water) ---')
mcc = cc.CCSD(mf_h2o).run()
mcc.solve_lambda()
print(get_esi_vals(mol_h2o, mcc, mf_h2o))
