import sys
import os
import numpy as np
from pyscf import gto, scf, mcscf, cc

sys.path.append(os.path.abspath('..'))
from esipy import ESI

atom_str = """
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

mol = gto.M(atom=atom_str, basis='sto-3g', spin=0, charge=0)
mol.build()

mf_scf = scf.RHF(mol).run()

def get_vals(partition, mf_obj, myhf_obj):
    ring = [1, 2, 3, 4, 5, 6]
    esi = ESI(mol=mol, mf=mf_obj, myhf=myhf_obj, rings=ring, partition=partition)
    aom, occ = esi.aom
    ind = esi.indicators[0]
    
    pop_at1 = np.einsum('i,ii->', occ, aom[0])
    lif = np.einsum('i,j,ij,ji->', occ**0.5, occ**0.5, aom[0], aom[0])
    lix = 0.5 * np.einsum('i,j,ij,ji->', occ, occ, aom[0], aom[0])
    
    lifs_sum = sum(np.einsum('i,j,ij,ji->', occ**0.5, occ**0.5, aom[k], aom[k]) for k in range(mol.natm))
    lixs_sum = 0.5 * sum(np.einsum('i,j,ij,ji->', occ, occ, aom[k], aom[k]) for k in range(mol.natm))
    difs_sum = sum(np.einsum('i,ii->', occ, aom[k]) - np.einsum('i,j,ij,ji->', occ**0.5, occ**0.5, aom[k], aom[k]) for k in range(mol.natm))
    dixs_sum = 0.5 * sum(sum(np.einsum('i,j,ij,ji->', occ, occ, aom[k], aom[l]) for l in range(mol.natm) if k != l) for k in range(mol.natm))

    return dict(
        exp_pop_atm1=round(float(pop_at1), 4),
        exp_lif_atm1=round(float(lif), 4),
        exp_lix_atm1=round(float(lix), 4),
        exp_dif_1_all=round(float(pop_at1 - lif), 4),
        exp_dix_1_all=round(float(0.5 * sum(np.einsum('i,j,ij,ji->', occ, occ, aom[0], aom[l]) for l in range(1, mol.natm))), 4),
        exp_lifs_sum=round(float(lifs_sum), 4),
        exp_lixs_sum=round(float(lixs_sum), 4),
        exp_difs_sum=round(float(difs_sum), 4),
        exp_dixs_sum=round(float(dixs_sum), 4),
        exp_iring=round(float(ind.iring), 6),
        exp_mci=round(float(ind.mci), 6),
        exp_av=round(float(ind.av1245), 3),
        exp_pdi=round(float(ind.pdi), 6),
        Nx=round(float(np.sum(occ)), 4)
    )

print('--- CAS IAO ---')
mc = mcscf.CASSCF(mf_scf, 6, 6).run()
print(get_vals('iao', mc, mf_scf))

print('\n--- CCSD IAO ---')
mycc = cc.CCSD(mf_scf).run()
mycc.solve_lambda()
print(get_vals('iao', mycc, mf_scf))
