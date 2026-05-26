import os
import sys
import numpy as np

# Ensure esipy is in path
sys.path.append(os.getcwd())

from esipy.readfchk import readfchk
from esipy import ESI
from pyscf import gto, scf

def run_pyscf_reference(mol_fchk):
    mol = gto.Mole()
    mol.atom = mol_fchk.atom
    mol.basis = mol_fchk.basis
    mol.charge = mol_fchk.charge
    mol.spin = mol_fchk.spin
    mol.cart = mol_fchk.cart
    mol.build()
    if mol.spin != 0:
         mf = scf.UHF(mol).run(verbose=0)
    else:
         mf = scf.RHF(mol).run(verbose=0)
    return mf

def main():
    base_dir = '../tests/FCHK/GAUSSIAN'
    filenames = [
        '1_benzene_spherical.fchk', '2_benzene_cartesian.fchk', '3_o2_triplet.fchk',
        '4_h2_oss.fchk', '5_high_l.fchk', '6_ecp.fchk', '7_rmp2.fchk',
        '8_cisd.fchk', '9_ccsd.fchk', '10_casscf_rest.fchk', '11_ump2.fchk',
        '12_casscf_unrest.fchk'
    ]
    partitions = ['mulliken', 'lowdin', 'nao', 'iao']
    
    print("EXPECTED_RESULTS = {")
    for f in filenames:
        path = os.path.join(base_dir, f)
        if not os.path.exists(path): continue
        
        mol_f, mf_f = readfchk(path)
        
        # Get NO occupations from FCHK object
        # mf_f._scf.mo_occ contains NO occupations if a density was found
        occ = mf_f.mo_occ
        if isinstance(occ, list):
             occ_sum = np.sum(occ[0]) + np.sum(occ[1])
             top_occ = occ[0][:3].tolist() # alpha
        else:
             occ_sum = np.sum(occ)
             top_occ = occ[:3].tolist()
             
        res = {
            'nelec': float(occ_sum),
            'top_occ': top_occ
        }
        print(f"    '{f}': {res},")
    print("}")

if __name__ == '__main__':
    main()
