import numpy as np
from pyscf import gto, scf, mcscf, mp
from esipy import ESI

def get_benzene_mol():
    mol = gto.M(atom='''
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
    ''', basis='sto-3g', spin=0, charge=0)
    mol.build()
    return mol

def get_refs(mf, mol, myhf=None):
    res = {}
    ring = [1, 2, 3, 4, 5, 6]
    for p in ['mulliken', 'lowdin', 'meta-lowdin', 'nao', 'iao']:
        esi = ESI(mol=mol, mf=mf, myhf=myhf, rings=ring, partition=p)
        aom, occ = esi.aom
        inds = esi.indicators[0]
        res[p] = {
            'exp_pop_atm1': float(np.einsum('i,ii->', occ, aom[0])),
            'exp_di12': float(2 * np.einsum('i,ij,j,ji->', np.sqrt(occ), aom[0], np.sqrt(occ), aom[1])),
            'exp_iring': float(inds.iring),
            'exp_mci': float(inds.mci),
            'exp_av': float(inds.av1245),
            'exp_pdi': float(inds.pdi)
        }
    return res

mol = get_benzene_mol()
myhf = scf.RHF(mol); myhf.init_guess='atom'; myhf.run()
print("REF5 =", get_refs(mcscf.CASSCF(myhf, 6, 6).run(), mol, myhf))
print("REF6 =", get_refs(mp.MP2(myhf).run(), mol, myhf))
