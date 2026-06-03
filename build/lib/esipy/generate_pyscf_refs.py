import numpy as np
from pyscf import gto, dft, mp, cc, mcscf, scf
import pickle
import os

# Set global VWN5 for Gaussian/Q-Chem B3LYP compatibility
dft.libxc.B3LYP_WITH_VWN5 = True

benzene_geo = """
C        0.000000     0.000000     1.393096
C        0.000000     1.206457     0.696548
C        0.000000     1.206457    -0.696548
C        0.000000     0.000000    -1.393096
C        0.000000    -1.206457    -0.696548
C        0.000000    -1.206457     0.696548
H        0.000000     0.000000     2.483127
H        0.000000     2.150450     1.241569
H        0.000000     2.150450    -1.241569
H        0.000000     0.000000    -2.483127
H        0.000000    -2.150450    -1.241569
H        0.000000    -2.150450     1.241569
"""
o2_geo = "O 0.0 0.0 0.0; O 0.0 0.0 1.207"
h2_geo = "H 0.0 0.0 0.0; H 0.0 0.0 3.0"
water_geo = "O 0.0 0.0 0.0; H 0.0 0.757 0.586; H 0.0 -0.757 0.586"
i2_geo = "I 0.0 0.0 0.0; I 0.0 0.0 2.66"

def get_indicators(mol, mf):
    from esipy import ESI
    from esipy.tools import find_di, find_ns, wf_type
    esi = ESI(mol=mol, mf=mf, partition='mulliken')
    aoms = esi.aom
    wf = wf_type(aoms)
    atoms = list(range(1, mol.natm+1))
    if wf == "unrest":
        pops = np.array(find_ns(atoms, aoms[0])) + np.array(find_ns(atoms, aoms[1]))
        di = find_di(aoms[0], 1, 2) + find_di(aoms[1], 1, 2) if mol.natm >= 2 else 0.0
    elif wf == "no":
        aom_list, occ = aoms
        pops = [np.sum([occ[i] * aom_list[a_idx][i,i] for i in range(len(occ))]) for a_idx in range(mol.natm)]
        di = 0.0
    else:
        pops = np.array(find_ns(atoms, aoms))
        di = find_di(aoms, 1, 2) if mol.natm >= 2 else 0.0
    return {"pops": np.asarray(pops), "di12": di}

def run_pyscf():
    refs = {}
    
    # 1. Benzene Spherical
    mol1 = gto.M(atom=benzene_geo, basis='cc-pVTZ', cart=False).build()
    mf1 = dft.RKS(mol1).set(xc='B3LYP').run()
    refs['1_benzene_spherical'] = {"e": mf1.e_tot, "ind": get_indicators(mol1, mf1)}
    
    # 2. Benzene Cartesian
    mol2 = gto.M(atom=benzene_geo, basis='cc-pVTZ', cart=True).build()
    mf2 = dft.RKS(mol2).set(xc='B3LYP').run()
    refs['2_benzene_cartesian'] = {"e": mf2.e_tot, "ind": get_indicators(mol2, mf2)}

    # 3. O2 Triplet
    mol3 = gto.M(atom=o2_geo, basis='6-31G(d)', spin=2).build()
    mf3 = dft.UKS(mol3).set(xc='B3LYP').run()
    refs['3_o2_triplet'] = {"e": mf3.e_tot, "ind": get_indicators(mol3, mf3)}

    # 4. H2 OSS (Broken Symmetry)
    mol4 = gto.M(atom=h2_geo, basis='6-31G(d)', spin=0).build()
    mf4 = dft.UKS(mol4).set(xc='B3LYP')
    dm = mf4.get_init_guess()
    dm[0, 0, 0] = 1.0; dm[0, 1, 1] = 0.0
    dm[1, 0, 0] = 0.0; dm[1, 1, 1] = 1.0
    mf4.level_shift = 0.2
    mf4.kernel(dm)
    refs['4_h2_oss'] = {"e": mf4.e_tot, "ind": get_indicators(mol4, mf4)}

    # 5. High L Water
    mol5 = gto.M(atom=water_geo, basis='cc-pVQZ').build()
    mf5 = dft.RKS(mol5).set(xc='B3LYP').run()
    refs['5_high_l'] = {"e": mf5.e_tot, "ind": get_indicators(mol5, mf5)}

    # 6. ECP I2
    mol6 = gto.M(atom=i2_geo, basis='lanl2dz', ecp='lanl2dz').build()
    mf6 = dft.RKS(mol6).set(xc='B3LYP').run()
    refs['6_ecp'] = {"e": mf6.e_tot, "ind": get_indicators(mol6, mf6)}

    # 7. RMP2 Water
    mol7 = gto.M(atom=water_geo, basis='cc-pVDZ').build()
    mf7_hf = scf.RHF(mol7).run()
    mmp2 = mp.MP2(mf7_hf).run()
    refs['7_rmp2'] = {"e": mmp2.e_tot, "ind": get_indicators(mol7, mmp2)}

    # 9. CCSD Water
    mcc = cc.CCSD(mf7_hf).run()
    refs['9_ccsd'] = {"e": mcc.e_tot, "ind": get_indicators(mol7, mcc)}
    
    # 10. CASSCF H2
    mol10 = gto.M(atom=h2_geo, basis='cc-pVDZ').build()
    mf10_hf = scf.RHF(mol10).run()
    mcas = mcscf.CASSCF(mf10_hf, 2, 2).run()
    refs['10_casscf_rest'] = {"e": mcas.e_tot, "ind": get_indicators(mol10, mcas)}

    return refs

if __name__ == "__main__":
    refs = run_pyscf()
    with open("pyscf_refs.pkl", "wb") as f:
        pickle.dump(refs, f)
    print("Persistent references generated.")
