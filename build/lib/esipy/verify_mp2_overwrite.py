import numpy as np
import os
import sys
# Add current directory to path
sys.path.append(os.getcwd())

from esipy.readfchk import readfchk
from pyscf import gto, scf, mp

def compare_mp2_density(fchk_path):
    print(f"\n--- Comparing PySCF MP2 vs FCHK SCF Density: {fchk_path} ---")
    # Our readfchk applies the correct Gaussian-style reordering
    mol_f, mf_f = readfchk(fchk_path)
    
    # 1. Run fresh PySCF MP2
    mol = gto.Mole()
    mol.atom = mol_f.atom
    mol.basis = mol_f.basis
    mol.charge = mol_f.charge
    mol.spin = mol_f.spin
    mol.cart = mol_f.cart
    mol.build()
    
    mf = scf.RHF(mol).run(verbose=0)
    mymp = mp.MP2(mf).run(verbose=0)
    dm_mp2_pyscf = mymp.make_rdm1()
    
    # 2. Get FCHK SCF density (which is now correctly permuted)
    dm_fchk = mf_f.make_rdm1()
    
    # Compare
    s = mol.intor_symmetric('int1e_ovlp')
    pop_pyscf = np.trace(dm_mp2_pyscf @ s)
    pop_fchk = np.trace(dm_fchk @ s)
    
    diff = np.linalg.norm(dm_mp2_pyscf - dm_fchk)
    print(f"  PySCF MP2 Pop: {pop_pyscf:.6f}")
    print(f"  FCHK SCF Pop:  {pop_fchk:.6f}")
    print(f"  DM Frobenius Diff: {diff:.6f}")
    
    if diff < 1e-3:
        print("  MATCH: The FCHK SCF label contains the MP2 density.")
    else:
        # Check if it matches RHF instead
        dm_rho = mf.make_rdm1()
        diff_rhf = np.linalg.norm(dm_rho - dm_fchk)
        print(f"  DM Diff from PySCF RHF: {diff_rhf:.6f}")
        if diff_rhf < 1e-3:
            print("  The FCHK SCF label contains the RHF density.")

# Q-Chem 7_rmp2.fchk is Water/cc-pVDZ
compare_mp2_density('../tests/FCHK/QCHEM/7_rmp2.fchk')
