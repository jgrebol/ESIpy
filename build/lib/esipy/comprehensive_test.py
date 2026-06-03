
import os
import numpy as np
from pyscf import gto, scf, lo
from readfchk import readfchk

def run_comprehensive_check(prog, filename):
    path = f"../tests/FCHK/{prog}/{filename}"
    if not os.path.exists(path):
        return
    
    print(f"\n{'='*80}")
    print(f" TESTING: {prog} / {filename}")
    print(f"{'='*80}")
    
    # 1. Load using readfchk
    mol_f, mf_f = readfchk(path)
    s_f = mf_f.get_ovlp()
    dm_f = mf_f.make_rdm1()
    
    # 2. Build Reference PySCF Mole
    mol_ref = gto.Mole()
    mol_ref.atom = mol_f.pyscf_mol.atom
    mol_ref.basis = mol_f.pyscf_mol.basis
    mol_ref.cart = mol_f.pyscf_mol.cart
    mol_ref.unit = 'Bohr'
    mol_ref.build()
    
    # Check Basis (mol._basis)
    # We compare the internal PySCF dictionaries
    basis_diff = False
    for sym in mol_ref._basis:
        b1 = mol_ref._basis[sym]
        b2 = mol_f.pyscf_mol._basis[sym]
        # Compare strings/structure roughly
        if str(b1) != str(b2):
            basis_diff = True
            break
    print(f" [BASIS] Identical to PySCF: {'YES' if not basis_diff else 'NO'}")
    
    # Check Overlap Matrix
    s_ref = mol_ref.intor_symmetric('int1e_ovlp')
    
    # We now compare DIRECTLY to the analytical overlap, as the code
    # has been updated to follow PySCF conventions.
    diff_s = np.linalg.norm(s_f - s_ref)
    print(f" [OVLP ] Norm Difference:   {diff_s:.2e}")
    
    # Check Density Matrix & Electrons
    if isinstance(dm_f, (list, tuple)) or dm_f.ndim == 3:
        nelec_f = np.trace((dm_f[0] + dm_f[1]) @ s_f)
    else:
        nelec_f = np.trace(dm_f @ s_f)
    
    print(f" [DM   ] Total Electrons:   {nelec_f:.6f} (Expected: {mol_ref.nelectron})")
    
    # Check Orthonormality (C.T @ S @ C)
    if isinstance(mf_f.mo_coeff, list):
        # UHF
        for i, c in enumerate(mf_f.mo_coeff):
            ortho = c.T @ s_f @ c
            err = np.linalg.norm(ortho - np.eye(c.shape[1]))
            print(f" [ORTHO] Spin {i} Error:      {err:.2e}")
    else:
        # RHF
        c = mf_f.mo_coeff
        ortho = c.T @ s_f @ c
        err = np.linalg.norm(ortho - np.eye(c.shape[1]))
        print(f" [ORTHO] Error:             {err:.2e}")

    # Check Transformation Matrix (Mulliken/Lowdin AOMs)
    try:
        from __init__ import ESI
        from tools import find_ns_rest, find_ns_unrest, find_ns_no, find_di_rest, find_di_unrest, find_di_no
        
        esi = ESI(mol=mol_f, mf=mf_f, partition='mulliken')
        aoms = esi.aom
        
        wf = getattr(esi, 'wf', 'rest') # Need to check how to get wf type
        # Or just use logic from aoms structure
        if isinstance(aoms, list) and len(aoms) == 2:
            if isinstance(aoms[1], np.ndarray):
                # NO
                pops = find_ns_no(list(range(1, mol_ref.natm+1)), aoms[0], aoms[1])
                di12 = find_di_no(aoms[0], aoms[1], 1, 2) if mol_ref.natm >= 2 else 0.0
            elif isinstance(aoms[1], list):
                # Unrestricted
                pops = find_ns_unrest(list(range(1, mol_ref.natm+1)), aoms[0], aoms[1])
                di12 = find_di_unrest(aoms[0], aoms[1], 1, 2) if mol_ref.natm >= 2 else 0.0
            else:
                # Should not happen with current ESI implementation
                pops = find_ns_rest(list(range(1, mol_ref.natm+1)), aoms)
                di12 = find_di_rest(aoms, 1, 2) if mol_ref.natm >= 2 else 0.0
        else:
            # Restricted
            pops = find_ns_rest(list(range(1, mol_ref.natm+1)), aoms)
            di12 = find_di_rest(aoms, 1, 2) if mol_ref.natm >= 2 else 0.0

        total_pop = sum(pops)
        print(f" [POP  ] Total (Mulliken):  {total_pop:.6f}")
        
        if mol_ref.natm >= 2:
            print(f" [DI   ] Atoms (1,2) DI:    {di12:.6f}")
            
    except Exception as e:
        print(f" [INDIC] Indicators Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    gauss_files = [
        '1_benzene_spherical.fchk',
        '2_benzene_cartesian.fchk',
        '3_o2_triplet.fchk',
        '5_high_l.fchk',
        '7_rmp2.fchk',
        '9_ccsd.fchk'
    ]
    qchem_files = [
        '1_benzene_spherical.fchk',
        '2_benzene_cartesian.fchk',
        '7_rmp2.fchk',
        '9_ccsd.fchk'
    ]
    
    for f in gauss_files:
        run_comprehensive_check('GAUSSIAN', f)
    for f in qchem_files:
        run_comprehensive_check('QCHEM', f)
