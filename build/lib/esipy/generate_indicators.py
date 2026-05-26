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
        '9_ccsd.fchk'
    ]
    partitions = ['mulliken', 'lowdin', 'nao', 'iao']
    
    print("EXPECTED_INDICATORS = {")
    for f in filenames:
        path = os.path.join(base_dir, f)
        if not os.path.exists(path): continue
        
        mol_f, _ = readfchk(path)
        mf_p = run_pyscf_reference(mol_f)
        
        # Determine rings for benzene
        rings = None
        if 'benzene' in f:
            rings = [[1,2,3,4,5,6]]
            
        system_res = {}
        for p in partitions:
            try:
                # Meta-lowdin not supported in UHF PySCF
                if p == 'meta-lowdin' and mol_f.spin != 0: continue
                
                esi = ESI(mol=mol_f, mf=mf_p, partition=p, rings=rings)
                
                # Get populations
                from esipy.tools import wf_type, find_di, find_di_no
                aoms = esi.aom
                wf = wf_type(aoms)
                if wf == "rest":
                    pops = [2 * np.trace(m) for m in aoms]
                elif wf == "unrest":
                    pops = [np.trace(a) + np.trace(b) for a, b in zip(aoms[0], aoms[1])]
                elif wf == "no":
                    aoms_list, occ_matrix = aoms
                    if occ_matrix.ndim == 1:
                        pops = [occ_matrix[i] * np.trace(aoms_list[i]) for i in range(len(aoms_list))]
                    else:
                        pops = [np.trace(occ_matrix @ m) for m in aoms_list]
                
                res = {
                    'pop_sum': float(np.sum(pops)),
                    'pop_atom1': float(pops[0])
                }
                
                # Indicators for rings
                if rings and len(esi.indicators) > 0:
                    ind = esi.indicators[0]
                    res['iring'] = float(ind.iring)
                    res['mci'] = float(ind.mci)
                    # DI between 1 and 2
                    if wf == "rest":
                        di12 = find_di(esi.aom, 1, 2)
                    elif wf == "unrest":
                        # For unrest, find_di takes a single spin component? 
                        # Let's check actual usage in indicators.py
                        # indicators.py uses 2*(find_di(aom_a, i, j) + find_di(aom_b, i, j))
                        di12 = 2 * (find_di(esi.aom[0], 1, 2) + find_di(esi.aom[1], 1, 2))
                    else:
                        di12 = find_di_no(esi.aom, 1, 2)
                    res['di12'] = float(di12)
                
                system_res[p] = res
            except Exception as e:
                import traceback
                system_res[p] = f"Error: {str(e)} \n {traceback.format_exc()}"
        
        print(f"    '{f}': {system_res},")
    print("}")

if __name__ == '__main__':
    main()
