import re

def fix():
    with open('readfchk.py', 'r') as f:
        content = f.read()

    # 1. Simplify get_ovlp to ALWAYS use PySCF (to match the analytical frame we now use)
    content = re.sub(r"def get_ovlp\(self\):.*?return self\.mol\.intor_symmetric\('int1e_ovlp'\)", 
                     r"def get_ovlp(self):\n        return self.mol.intor_symmetric('int1e_ovlp')", 
                     content, flags=re.DOTALL)

    # 2. Refine MeanField2.__init__ for UHF Canonical MOs
    # We want to be really clean.
    
    clean_init = """
        # --- UHF / Open-Shell Reading Logic ---
        # Prioritize Canonical MO coefficients (Alpha/Beta) for unrestricted calculations
        mo_a_flat = read_list_from_fchk('Alpha MO coefficients', path)
        mo_b_flat = read_list_from_fchk('Beta MO coefficients', path)
        
        is_uhf = (len(mo_b_flat) > 0) or unrestricted
        
        if is_uhf and len(mo_a_flat) > 0 and len(mo_b_flat) > 0:
            print(" | Using Canonical UHF MO coefficients")
            mo_arr_a = np.array(mo_a_flat, dtype=float).reshape(self.nummo, self.nao).T
            mo_arr_b = np.array(mo_b_flat, dtype=float).reshape(self.nummo, self.nao).T
            self.mo_coeff = [standardize_mat(mo_arr_a), standardize_mat(mo_arr_b)]
            self.mo_occ = [np.zeros(self.nummo), np.zeros(self.nummo)]
            self.mo_occ[0][:self.nalpha] = 1.0
            self.mo_occ[1][:self.nbeta] = 1.0
        else:
            # Try reading Total Density first (for post-HF or if MOs missing)
            d_labels = [
                ('Total CI Rho(1) Density', 'Spin CI Rho(1) Density'),
                ('Total CI Density', 'Spin CI Density'),
                ('Total CC Density', 'Spin CC Density'),
                ('Total MP2 Density', 'Spin MP2 Density'),
                ('Total SCF Density', 'Spin SCF Density'),
            ]
            
            found_density = False
            for t_lbl, s_lbl in d_labels:
                dt_flat = read_list_from_fchk(t_lbl, path)
                if len(dt_flat) > 0:
                    print(f" | Found density: {t_lbl}")
                    found_density = True
                    
                    n = self.nao
                    if len(dt_flat) == n*(n+1)//2: # Triangular
                        mat_gau = np.zeros((n, n))
                        mat_gau[np.tril_indices(n)] = dt_flat
                        mat_gau = mat_gau + mat_gau.T - np.diag(np.diag(mat_gau))
                    else: # Full
                        mat_gau = np.array(dt_flat).reshape(n, n)
                    
                    dt = standardize_mat(mat_gau)
                    
                    # Convert Total Density to NO representation (Restricted-like)
                    s_ovlp = self.mol.intor_symmetric('int1e_ovlp')
                    from scipy.linalg import eigh
                    occ, coeff = eigh(s_ovlp @ dt @ s_ovlp, b=s_ovlp)
                    idx_no = np.argsort(occ)[::-1]
                    self.mo_occ = occ[idx_no]
                    self.mo_coeff = coeff[:, idx_no]
                    self.mo_occ[self.mo_occ < 1e-12] = 0.0
                    self._scf.make_rdm1 = lambda *args, **kwargs: dt
                    break
                    
            if not found_density:
                # Fallback to standard Alpha MOs
                if len(mo_a_flat) > 0:
                    print(" | Using Fallback RHF/ROHF MO coefficients")
                    mo_arr_a = np.array(mo_a_flat, dtype=float).reshape(self.nummo, self.nao).T
                    self.mo_coeff = standardize_mat(mo_arr_a)
                    self.mo_occ = np.zeros(self.nummo)
                    nocc = (self.nalpha + self.nbeta) // 2
                    self.mo_occ[:nocc] = 2.0
                else:
                    raise RuntimeError('No MO coefficients or Density found in FCHK')

        self._scf.mo_coeff = self.mo_coeff
        self._scf.mo_occ = self.mo_occ
"""

    # Replace the block
    pattern = re.compile(r"# --- NEW LOGIC.*?self\._scf\.mo_occ = self\.mo_occ", re.DOTALL)
    content = pattern.sub(clean_init, content)
    
    with open('readfchk.py', 'w') as f:
        f.write(content)

if __name__ == "__main__":
    fix()
