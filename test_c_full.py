import sys
sys.path.insert(0, "/home/joan/PycharmProjects/ESIpy")
from pyscf import gto, scf
from esipy.make_aoms import make_aoms

mol = gto.M(atom='Li 0 0 0; H 0 0 1.6', basis='cc-pvdz')
mf = scf.RHF(mol).run(verbose=0)

import esipy.make_aoms as ma
original_get_iao = ma.get_iao_aoms

def spy_get_iao_aoms(p_type, c, current_mf, w_override=None, current_myhf=None, c_full=None):
    print(f"c shape: {c.shape}")
    print(f"c_full is None? {c_full is None}")
    if c_full is not None:
        print(f"c_full shape: {c_full.shape}")
    proj_c = c_full if c_full is not None else c
    print(f"proj_c shape: {proj_c.shape}")
    return original_get_iao(p_type, c, current_mf, w_override, current_myhf, c_full)

ma.get_iao_aoms = spy_get_iao_aoms
make_aoms(mol, mf, 'iao')
