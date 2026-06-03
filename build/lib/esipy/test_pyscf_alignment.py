import numpy as np
from pyscf import gto
from readfchk import read_list_from_fchk, readfchk

def test_alignment(prog, filename):
    path = f"../tests/FCHK/{prog}/{filename}"
    print(f"\n=== Testing Alignment to PySCF Convention: {prog} {filename} ===")
    
    # 1. Build PySCF reference (Unnormalized Cartesian)
    mol_f, _ = readfchk(path)
    mol_p = gto.Mole()
    mol_p.atom = mol_f.pyscf_mol.atom
    mol_p.basis = mol_f.pyscf_mol.basis
    mol_p.cart = True
    mol_p.unit = 'Bohr'
    mol_p.build()
    S_p = mol_p.intor('int1e_ovlp') # The "Analytical" Overlap
    
    # 2. Get RAW MO coefficients from FCHK (Alpha)
    # We need them BEFORE any ESIpy scaling.
    mo_flat = read_list_from_fchk('Alpha MO coefficients', path)
    nbf = mol_f.numao
    nif = mol_f.nummo
    C_raw = np.array(mo_flat).reshape(nif, nbf).T
    
    # 3. Apply ONLY reordering to C_raw (no scaling yet)
    from esipy.tools import permute_aos_rows
    C_ordered = permute_aos_rows(C_raw, mol_f)
    
    # 4. Check Orthonormality with S_p (Analytical)
    # If Gaussian is Unit-Normalized and PySCF is Analytical, 
    # then C_ordered.T @ S_p @ C_ordered will NOT be Identity.
    overlap_raw = C_ordered.T @ S_p @ C_ordered
    print(f"  Raw Orthonormality Error (max diff from I): {np.max(np.abs(overlap_raw - np.eye(nif))):.4e}")
    
    # 5. Apply "PySCF-style" Scaling
    # Convention: C_pyscf = C_unit / sqrt(diag(S_pyscf))
    # This assumes Gaussian's C is unit-normalized (diagonal of S_gau = 1)
    s_diag = np.diag(S_p)
    v_pyscf = 1.0 / np.sqrt(s_diag)
    C_pyscf = (C_ordered.T * v_pyscf).T
    
    overlap_scaled = C_pyscf.T @ S_p @ C_pyscf
    print(f"  Scaled Orthonormality Error (max diff from I): {np.max(np.abs(overlap_scaled - np.eye(nif))):.4e}")
    
    # 6. Verify Electron Trace
    # P_pyscf = C_pyscf @ n @ C_pyscf.T
    # Trace(P_pyscf @ S_p) should be exactly Nelec
    nalpha = mol_f.fchk.nalpha
    occ = np.zeros(nif)
    occ[:nalpha] = 1.0 # Assuming RHF/Alpha
    P_pyscf = (C_pyscf * occ) @ C_pyscf.T
    nelec = np.trace(P_pyscf @ S_p)
    print(f"  Trace(P_scaled @ S_pyscf): {nelec:.6f} (Target: {nalpha})")

test_alignment('GAUSSIAN', '2_benzene_cartesian.fchk')
