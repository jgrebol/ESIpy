import re

def apply_fix():
    with open('readfchk.py', 'r') as f:
        content = f.read()

    # Define the new logic that PRIORITIZES Canonical MOs for UHF
    new_init_logic = """
        # --- NEW LOGIC: Prioritize Canonical MOs for UHF ---
        
        # 1. Determine if it should be unrestricted
        beta_mo_exists = len(read_list_from_fchk('Beta MO coefficients', path)) > 0
        is_unrest = unrestricted or beta_mo_exists
        
        mo_read_success = False
        if is_unrest:
            try:
                mo_a_flat = read_list_from_fchk('Alpha MO coefficients', path)
                mo_b_flat = read_list_from_fchk('Beta MO coefficients', path)
                if len(mo_a_flat) > 0 and len(mo_b_flat) > 0:
                    print(" | Using Canonical UHF MO coefficients")
                    mo_arr_a = np.array(mo_a_flat, dtype=float).reshape(self.nummo, self.nao).T
                    mo_arr_b = np.array(mo_b_flat, dtype=float).reshape(self.nummo, self.nao).T
                    self.mo_coeff = [standardize_mo(mo_arr_a), standardize_mo(mo_arr_b)]
                    self.mo_occ = [np.zeros(self.nummo), np.zeros(self.nummo)]
                    self.mo_occ[0][:self.nalpha] = 1.0
                    self.mo_occ[1][:self.nbeta] = 1.0
                    mo_read_success = True
                    self._scf = scf.UHF(self.mol)
                    self.__name__ = "UHF"
            except Exception as e:
                print(f" | Warning: Could not read Canonical UHF MOs: {e}")

        if not mo_read_success:
            # Try reading post-HF densities (Natural Orbitals)
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
                    print(f" | Found density: {t_lbl} (Ignoring Spin Density)")
                    found_density = True
                    n = self.nao
                    if len(dt_flat) == n*(n+1)//2:
                        mat_gau = np.zeros((n, n))
                        mat_gau[np.tril_indices(n)] = dt_flat
                        mat_gau = mat_gau + mat_gau.T - np.diag(np.diag(mat_gau))
                    else:
                        mat_gau = np.array(dt_flat).reshape(n, n)
                    
                    dt = standardize_dm(mat_gau)
                    
                    # NO representation
                    s_ovlp = self.mol.intor_symmetric('int1e_ovlp')
                    from scipy.linalg import eigh
                    occ, coeff = eigh(s_ovlp @ dt @ s_ovlp, b=s_ovlp)
                    idx_no = np.argsort(occ)[::-1]
                    self.mo_occ = occ[idx_no]
                    self.mo_coeff = coeff[:, idx_no]
                    self.mo_occ[self.mo_occ < 1e-12] = 0.0

                    self._scf = scf.RHF(self.mol) # RHF object for NOs
                    self.__name__ = "RHF"
                    self._scf.make_rdm1 = lambda *args, **kwargs: dt
                    break

            if not found_density:
                # Last resort: Restricted Alpha MOs
                mo_a_flat = read_list_from_fchk('Alpha MO coefficients', path)
                if len(mo_a_flat) > 0:
                    print(" | Using Fallback RHF/ROHF MO coefficients")
                    mo_arr_a = np.array(mo_a_flat, dtype=float).reshape(self.nummo, self.nao).T
                    self.mo_coeff = standardize_mo(mo_arr_a)
                    self.mo_occ = np.zeros(self.nummo)
                    nocc = (self.nalpha + self.nbeta) // 2
                    self.mo_occ[:nocc] = 2.0
                    self._scf = scf.RHF(self.mol)
                    self.__name__ = "RHF"
                else:
                    raise RuntimeError('No MO coefficients or Density found in FCHK')

        self._scf.mo_coeff = self.mo_coeff
        self._scf.mo_occ = self.mo_occ
"""

    # Replace the block from # --- UHF / Open-Shell Reading Logic --- to self._scf.mo_occ = self.mo_occ
    pattern = re.compile(r"# --- UHF / Open-Shell Reading Logic ---.*?self\._scf\.mo_occ = self\.mo_occ", re.DOTALL)
    content = pattern.sub(new_init_logic, content)

    with open('readfchk.py', 'w') as f:
        f.write(content)

if __name__ == "__main__":
    apply_fix()
