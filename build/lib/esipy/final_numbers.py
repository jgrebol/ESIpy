import os
import sys
import numpy as np
# Add current directory to path
sys.path.append(os.getcwd())

from esipy.readfchk import readfchk
from esipy import ESI
from pyscf import gto, scf

def get_pops(esi):
    from esipy.tools import wf_type
    aoms = esi.aom
    wf = wf_type(aoms)
    if wf == "rest":
        return np.array([2 * np.trace(m) for m in aoms])
    elif wf == "unrest":
        return np.array([np.trace(a) + np.trace(b) for a, b in zip(aoms[0], aoms[1])])
    elif wf == "no":
        aoms_list, occ_matrix = aoms
        if occ_matrix.ndim == 1:
            return np.array([np.einsum('i,ii->', occ_matrix, m) for m in aoms_list])
        else:
            return np.array([np.trace(occ_matrix @ m) for m in aoms_list])
    return np.array([])

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
    systems = [
        ('1_benzene_spherical.fchk', 42.0, [[1,2,3,4,5,6]]),
        ('2_benzene_cartesian.fchk', 42.0, [[1,2,3,4,5,6]]),
        ('3_o2_triplet.fchk', 16.0, None),
        ('9_ccsd.fchk', 10.0, None)
    ]
    
    partitions = ['mulliken', 'lowdin', 'nao', 'iao']
    
    print(f"{'System':<25} | {'Part':<8} | {'PySCF':<10} | {'GAUSSIAN':<10} | {'Q-CHEM':<10}")
    print("-" * 75)

    for filename, nelec, rings in systems:
        # 1. Load FCHKs
        path_g = os.path.join('../tests/FCHK/GAUSSIAN', filename)
        path_q = os.path.join('../tests/FCHK/QCHEM', filename)
        
        mol_g, mf_g = readfchk(path_g)
        mol_q, mf_q = readfchk(path_q)
        
        # 2. Run PySCF reference
        mf_p = run_pyscf_reference(mol_g)
        
        for p in partitions:
            # Skip partitions not compatible with current reference logic (UHF meta-lowdin)
            if p == 'meta-lowdin' and mol_g.spin != 0: continue
            
            # Get populations
            esi_p = ESI(mol=mol_g, mf=mf_p, partition=p, rings=rings)
            esi_g = ESI(mol=mol_g, mf=mf_g, partition=p, rings=rings)
            esi_q = ESI(mol=mol_q, mf=mf_q, partition=p, rings=rings)
            
            pop_p = np.sum(get_pops(esi_p))
            pop_g = np.sum(get_pops(esi_g))
            pop_q = np.sum(get_pops(esi_q))
            
            print(f"{filename:<25} | {p:<8} | {pop_p:10.4f} | {pop_g:10.4f} | {pop_q:10.4f}")
            
            # If benzene, show Iring too
            if rings and p == 'mulliken':
                ir_p = esi_p.indicators[0].iring
                ir_g = esi_g.indicators[0].iring
                ir_q = esi_q.indicators[0].iring
                print(f"{'  (Iring)':<25} | {'-':<8} | {ir_p:10.6f} | {ir_g:10.6f} | {ir_q:10.6f}")

if __name__ == '__main__':
    main()
