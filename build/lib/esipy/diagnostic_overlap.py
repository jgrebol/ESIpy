import numpy as np
from pyscf import gto
from readfchk import read_list_from_fchk, readfchk

def diagnose(prog, filename):
    path = f"../tests/FCHK/{prog}/{filename}"
    print(f"\n=== Diagnostic: {prog} {filename} ===")
    
    # 1. Read Raw Overlap from FCHK (if available, mostly Q-Chem)
    ovlp_flat = read_list_from_fchk("Overlap Matrix", path)
    
    # 2. Build PySCF reference
    # For Benzene Cartesian
    mol_f, _ = readfchk(path)
    mol_p = gto.Mole()
    mol_p.atom = mol_f.pyscf_mol.atom
    mol_p.basis = mol_f.pyscf_mol.basis
    mol_p.cart = True
    mol_p.unit = 'Bohr'
    mol_p.build()
    S_p = mol_p.intor('int1e_ovlp')
    
    if not ovlp_flat:
        print("  [INFO] No 'Overlap Matrix' block in FCHK. This is typical for Gaussian.")
        # For Gaussian, we can't compare the matrix directly from the file,
        # but we can look at the diagonal of S_p vs what ESIpy tries to do.
        print("  PySCF S_p diag (first 20):", np.round(np.diag(S_p)[:20], 4))
        return

    n = mol_p.nao
    S_f = np.zeros((n, n))
    if len(ovlp_flat) == n*(n+1)//2:
        idx = 0
        for i in range(n):
            for j in range(i+1):
                S_f[i, j] = S_f[j, i] = ovlp_flat[idx]
                idx += 1
    elif len(ovlp_flat) == n*n:
        S_f = np.array(ovlp_flat).reshape(n, n)
    
    print("  PySCF S_p diag (first 20):", np.round(np.diag(S_p)[:20], 4))
    print("  FCHK  S_f diag (first 20):", np.round(np.diag(S_f)[:20], 4))
    
    # Compare ratios for a D-shell (if index known)
    # For Benzene cc-pVTZ Cartesian: C is 1s, 2s, 3s, 4s, 2p, 3p, 4p, 3d, 4d, 4f
    # Shell types: 0, 0, 0, 0, 1, 1, 1, 2, 2, 3
    # AO counts: 1, 1, 1, 1, 3, 3, 3, 6, 6, 10
    # D-shell starts at index 13
    d_p = np.diag(S_p)[13:19]
    d_f = np.diag(S_f)[13:19]
    print("  D-shell Diag (PySCF):", d_p)
    print("  D-shell Diag (FCHK): ", d_f)
    print("  PySCF Ratios (XX:XY):", d_p[0]/d_p[1] if d_p[1] != 0 else "N/A")
    print("  FCHK  Ratios (XX:XY):", d_f[0]/d_f[1] if d_f[1] != 0 else "N/A")

diagnose('GAUSSIAN', '2_benzene_cartesian.fchk')
diagnose('QCHEM', '2_benzene_cartesian.fchk')
