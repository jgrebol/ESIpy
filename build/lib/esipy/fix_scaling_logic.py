import re

def fix():
    with open('readfchk.py', 'r') as f:
        content = f.read()

    # standardize_mo should only permute, because permute_aos_rows ALREADY scales
    # standardize_dm should permute and scale both sides.
    # But permute_aos_rows scales once.
    # So if we call it twice (once for rows, once for cols), we get (1/Sdiag)*(1/Sdiag) = 1/Sdiag^2? No.
    # Gaussian D = C n C^T. 
    # C_pyscf = C_gau / sqrt(Sdiag)
    # D_pyscf = (C_gau / sqrt(Sdiag)) n (C_gau / sqrt(Sdiag))^T = D_gau / Sdiag
    
    # permute_aos_rows(mo) -> C_gau[p] / sqrt(Sdiag)  (Correct)
    # permute_aos_rows(dm) -> D_gau[p, :] / sqrt(Sdiag)
    # permute_aos_rows(dm.T) -> D_gau[p, p] / (sqrt(Sdiag)*sqrt(Sdiag)) = D_gau[p,p] / Sdiag (Correct)

    clean_scaling = """
        # Get the analytical overlap matrix from PySCF for the reconstructed basis
        S_raw = self.mol.intor_symmetric('int1e_ovlp')

        def standardize_mo(mo_arr):
            from esipy.tools import permute_aos_rows
            # permute_aos_rows reorders and divides by sqrt(diag(S_pyscf)) for L>=2
            return permute_aos_rows(mo_arr, self.mole2)

        def standardize_dm(dm_arr):
            from esipy.tools import permute_aos_rows
            # Apply to rows then columns to get total 1/diag(S_pyscf) scaling
            dm_p = permute_aos_rows(dm_arr, self.mole2)
            return permute_aos_rows(dm_p.T, self.mole2).T
"""

    pattern_block = re.compile(r"(\s+)# Get the analytical overlap matrix.*?return dm_p \* np\.outer\(v_align, v_align\)", re.DOTALL)
    content = pattern_block.sub(clean_scaling, content)

    with open('readfchk.py', 'w') as f:
        f.write(content)

if __name__ == "__main__":
    fix()
