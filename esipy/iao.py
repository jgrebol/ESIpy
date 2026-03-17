import numpy as np
import scipy.linalg
from pyscf import gto, scf
from pyscf.data import elements
from pyscf.lo import orth

def spherical_average(mol, ia, mat, overlap):
    """
    Performs spherical averaging of a property matrix for atom 'ia'.
    Returns eigenvalues and eigenvectors grouped by shell.
    """
    # Identify shells belonging to this atom
    all_l = [mol.bas_angular(ib) for ib in range(mol.nbas)]
    all_atoms = [mol.bas_atom(ib) for ib in range(mol.nbas)]
    
    atom_shell_indices = [ib for ib, at in enumerate(all_atoms) if at == ia]
    l_vals = [all_l[ib] for ib in atom_shell_indices]
    unique_l = sorted(list(set(l_vals)))
    
    ao_loc = mol.ao_loc_nr()
    atom_start = mol.aoslice_by_atom()[ia, 2]
    
    all_w = []
    all_c = np.zeros_like(mat)
    l_map = [] 
    current_col = 0
    
    for l in unique_l:
        # Number of components per contracted shell
        degen = 2 * l + 1 if not mol.cart else (l + 1) * (l + 2) // 2
        
        # Collect all contracted shells of this L
        # Note: one 'ib' in mol._bas can have multiple contracted shells (n_contr > 1)
        subshells = []
        for ib in atom_shell_indices:
            if mol.bas_angular(ib) == l:
                n_contr = mol.bas_nctr(ib)
                p0 = ao_loc[ib] - atom_start
                for k in range(n_contr):
                    # Index of first AO of this contracted subshell
                    subshells.append(p0 + k * degen)
        
        n_shells = len(subshells)
        mat_red = np.zeros((n_shells, n_shells))
        ovlp_red = np.zeros((n_shells, n_shells))
        
        for i, p0 in enumerate(subshells):
            for j, q0 in enumerate(subshells):
                # We average over the 'degen' components
                # The components are usually ordered: (s0_m0, s0_m1, ..., s1_m0, s1_m1, ...)
                # WAIT: in PySCF, if n_contr > 1, the order is (s0_m0, s1_m0, ..., s0_m1, s1_m1, ...)
                # Let's verify this. Actually, standard PySCF AO order for n_contr > 1:
                # [shell0_comp0, shell1_comp0, ..., shell0_comp1, shell1_comp1, ...]
                # So the 'degen' components of shell 'i' are NOT contiguous!
                
                # Let's find the indices of the 'degen' components for subshell i
                # ib is the parent shell index
                # i is the subshell index within the L-group
                
                # Correction: loop over components m
                val_mat = 0.0
                val_ovlp = 0.0
                for m in range(degen):
                    # In PySCF, the AO index for (subshell i, component m) is p0 + m*n_shells_in_ib?
                    # No, let's use a safer approach:
                    pass
        
        # Simplified approach: assume n_contr = 1 for now or handle the PySCF mapping
        # Most modern bases in PySCF are n_contr=1 expanded.
        # If n_contr > 1, we'll just use the trace over the whole block.
        for i, ib in enumerate(shell_indices_l := [ib for ib in atom_shell_indices if mol.bas_angular(ib) == l]):
            for j, jb in enumerate(shell_indices_l):
                p0, p1 = ao_loc[ib] - atom_start, ao_loc[ib+1] - atom_start
                q0, q1 = ao_loc[jb] - atom_start, ao_loc[jb+1] - atom_start
                
                # Average the block (Trace / dimensionality)
                # This works correctly even if n_contr > 1, as it averages all functions of this L
                mat_red_block = mat[p0:p1, q0:q1]
                ovlp_red_block = overlap[p0:p1, q0:q1]
                
                # mat_red_block is (degen*n_contr_i) x (degen*n_contr_j)
                # We want to extract the (n_contr_i x n_contr_j) radial part
                ni, nj = mol.bas_nctr(ib), mol.bas_nctr(jb)
                m_sub = np.zeros((ni, nj))
                o_sub = np.zeros((ni, nj))
                for ki in range(ni):
                    for kj in range(nj):
                        # Sum over components m
                        s = 0.0
                        so = 0.0
                        for m in range(degen):
                            # PySCF ordering: component m is at p0 + m*ni + ki
                            idx_i = ki + m * ni
                            idx_j = kj + m * nj
                            s += mat_red_block[idx_i, idx_j]
                            so += ovlp_red_block[idx_i, idx_j]
                        m_sub[ki, kj] = s / degen
                        o_sub[ki, kj] = so / degen
                
                # Now we need to place m_sub into the larger mat_red
                # We'll re-structure mat_red to be N_contracted_shells total
                pass

    # Actually, let's use a simpler, more robust logic for mat_red
    # 1. Expand shells so each ib has n_contr=1
    # 2. Compute mat_red
    
    # 1. Identify all subshells (L, atom, indices in local atom matrix)
    atom_subshells = []
    for ib in atom_shell_indices:
        l = mol.bas_angular(ib)
        ni = mol.bas_nctr(ib)
        degen = 2*l+1 if not mol.cart else (l+1)*(l+2)//2
        p0 = ao_loc[ib] - atom_start
        for k in range(ni):
            indices = [p0 + k + m*ni for m in range(degen)]
            atom_subshells.append({'l': l, 'indices': indices})
            
    unique_l = sorted(list(set(s['l'] for s in atom_subshells)))
    
    all_w = []
    all_c = np.zeros((mat.shape[0], mat.shape[0]))
    l_map = []
    current_col = 0
    
    for l in unique_l:
        l_subshells = [s for s in atom_subshells if s['l'] == l]
        n = len(l_subshells)
        mat_red = np.zeros((n, n))
        ovlp_red = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                idx_i = l_subshells[i]['indices']
                idx_j = l_subshells[j]['indices']
                # Average over components
                s = 0.0
                so = 0.0
                for m in range(len(idx_i)):
                    s += mat[idx_i[m], idx_j[m]]
                    so += overlap[idx_i[m], idx_j[m]]
                mat_red[i, j] = s / len(idx_i)
                ovlp_red[i, j] = so / len(idx_i)
        
        # Solve GEVP
        try:
            e, v = scipy.linalg.eigh(ovlp_red)
            e[e < 1e-14] = 1e-14
            s_inv_half = v @ np.diag(1.0 / np.sqrt(e)) @ v.T
            w, c_prime = scipy.linalg.eigh(s_inv_half @ mat_red @ s_inv_half)
            c_red = s_inv_half @ c_prime
        except:
            w, c_red = scipy.linalg.eigh(mat_red)
            
        # Sort descending
        idx = np.argsort(w)[::-1]
        w, c_red = w[idx], c_red[:, idx]
        
        print(f"L={l} Shell Occupations (Averaged): {w}")
        
        degen = len(l_subshells[0]['indices'])
        for iw in range(n):
            val = w[iw]
            for m in range(degen):
                v_full = np.zeros(mat.shape[0])
                for i in range(n):
                    # Radial coefficient c_red[i, iw] for subshell i, component m
                    v_full[l_subshells[i]['indices'][m]] = c_red[i, iw]
                
                # Norm
                norm = np.sqrt(v_full.T @ overlap @ v_full)
                if norm > 1e-14: v_full /= norm
                
                all_w.append(val)
                l_map.append(l)
                all_c[:, current_col] = v_full
                current_col += 1
                
    return np.array(all_w), all_c, np.array(l_map)

def get_num_minbas_per_l(sym, polarized=False):
    z = elements.charge(sym)
    if z <= 2:   d = {0: 1}  # 1s
    elif z <= 10:  d = {0: 2, 1: 1}  # 1s, 2s, 2p
    elif z <= 18:  d = {0: 3, 1: 2}  # 3s, 3p
    elif z <= 36:  d = {0: 4, 1: 3, 2: 1}  # 4s, 4p, 3d
    elif z <= 54:  d = {0: 5, 1: 4, 2: 2}  # 5s, 5p, 4d
    elif z <= 86:  d = {0: 6, 1: 5, 2: 3, 3: 1}
    elif z <= 118: d = {0: 7, 1: 6, 2: 4, 3: 2}
    else: raise NotImplementedError(f"Minimal basis not defined for: {sym}")
    if polarized:
        if z <= 2: d[1] = 1
        elif z <= 10: d[2] = 1
        elif z <= 18: d[2] = 1
        elif z <= 36: d[3] = 1
        elif z <= 54: d[3] = 1
    return d

def get_num_minbas_ao(mol, ia, polarized=False):
    sym = mol.atom_pure_symbol(ia)
    target_l = get_num_minbas_per_l(sym, polarized=polarized)
    total = 0
    # Find degeneracies in the working basis
    for l, count in target_l.items():
        degen = 2*l+1
        for ib in range(mol.nbas):
            if mol.bas_atom(ib) == ia and mol.bas_angular(ib) == l:
                degen = (mol.ao_loc_nr()[ib+1] - mol.ao_loc_nr()[ib]) // mol.bas_nctr(ib)
                break
        total += count * degen
    return total

def get_reference_basis_dict(mol, source_basis='minao', pol_basis=None, x=1.0):
    def get_minimal_part(sym, source_basis_name):
        basis_source = gto.basis.load(source_basis_name, sym)
        target_l = get_num_minbas_per_l(sym, polarized=False)
        new_basis_atom, l_counts = [], {}
        for shell in basis_source:
            l = shell[0]
            count = l_counts.get(l, 0)
            if count < target_l.get(l, 0):
                new_basis_atom.append(shell)
                l_counts[l] = count + 1
        return new_basis_atom, target_l

    def get_polarization_part(sym, pol_basis_name, target_l, x=1.0):
        if pol_basis_name == 'working':
            basis_pol = mol._basis[sym]
        elif isinstance(pol_basis_name, str):
            basis_pol = gto.basis.load(pol_basis_name, sym)
        else: return []
        target_pol_l = get_num_minbas_per_l(sym, polarized=True)
        new_pol_atom, l_counts = [], {}
        for shell in basis_pol:
            l = shell[0]
            if l in target_pol_l and l not in target_l:
                if l_counts.get(l, 0) < target_pol_l[l]:
                    prim = shell[1]
                    scaled_exp = prim[0] * x
                    new_pol_atom.append([l, [scaled_exp, 1.0]])
                    l_counts[l] = l_counts.get(l, 0) + 1
        return new_pol_atom

    ref_basis = {}
    for ia in range(mol.natm):
        sym = mol.atom_pure_symbol(ia)
        if sym not in ref_basis:
            min_basis, target_l = get_minimal_part(sym, source_basis)
            pol_basis_list = get_polarization_part(sym, pol_basis, target_l, x=x)
            ref_basis[sym] = min_basis + pol_basis_list
    return ref_basis

def reference_mol(mol, polarized=False, pol_basis=None, source_basis='minao', x=1.0):
    from pyscf.lo import iao as pyscf_iao
    if hasattr(mol, 'pyscf_mol'): mol = mol.pyscf_mol
    if polarized and pol_basis is None: pol_basis = 'minao'
    elif not polarized: pol_basis = None
    ref_basis = get_reference_basis_dict(mol, source_basis=source_basis, pol_basis=pol_basis, x=x)
    return pyscf_iao.reference_mol(mol, minao=ref_basis)

def get_effaos(mol, mf, free_atom=True, mode=None, polarized=False):
    pmol = reference_mol(mol, polarized=polarized)
    minbas_total = pmol.nao
    veps_block = np.zeros((mol.nao, minbas_total))
    vaps_diag = []
    S_mol = mol.intor("int1e_ovlp")
    if "U" in mf.__class__.__name__:
        dm = mf.make_rdm1(ao_repr=True); P_mol = dm[0] + dm[1] if dm.ndim == 3 else dm
    else:
        P_mol = mf.make_rdm1(ao_repr=True)

    if not free_atom:
        if mode in ["lowdin", "meta-lowdin"]:
            if mode == "lowdin": T = orth.lowdin(S_mol)
            else:
                from pyscf.lo.orth import orth_ao
                T = orth_ao(mf, 'meta_lowdin', pre_orth_ao="MINAO")
            T_inv = scipy.linalg.sqrtm(S_mol)
            P_mol = T_inv @ P_mol @ T_inv
            S_mol = np.eye(mol.nao)
        elif mode == "gross":
            P_mol = (P_mol @ S_mol + S_mol @ P_mol) / 2
            S_mol = np.eye(mol.nao)

    aoslices = mol.aoslice_by_atom()
    col_idx = 0
    atom_spins = {'H':1,'He':0,'Li':1,'Be':0,'B':1,'C':2,'N':3,'O':2,'F':1,'Ne':0,'Na':1,'Mg':0,'Al':1,'Si':2,'P':3,'S':2,'Cl':1,'Ar':0}

    for ia in range(mol.natm):
        sym = mol.atom_pure_symbol(ia)
        p0, p1 = aoslices[ia, 2], aoslices[ia, 3]
        target_l_counts = get_num_minbas_per_l(sym, polarized=polarized)
        n_target = get_num_minbas_ao(mol, ia, polarized=polarized)

        if free_atom:
            mol_at = gto.Mole(); mol_at.atom = f"{sym} 0 0 0"; mol_at.basis = {sym: mol._basis[sym]}
            mol_at.spin = atom_spins.get(sym, 0); mol_at.cart = mol.cart; mol_at.build()
            mf_at = scf.KS(mol_at) if "dft" in mf.__module__ else scf.HF(mol_at)
            if hasattr(mf, 'xc'): mf_at.xc = mf.xc
            mf_at.kernel()
            dm_at = mf_at.make_rdm1()
            P_at = dm_at[0] + dm_at[1] if dm_at.ndim == 3 else dm_at
            S_at = mf_at.get_ovlp()
            mat_block = S_at @ P_at @ S_at
            ovlp_block = S_at
            w, c, l_map = spherical_average(mol_at, 0, mat_block, ovlp_block)
        else:
            if mode == "net":
                Sa, Pa = S_mol[p0:p1, p0:p1], P_mol[p0:p1, p0:p1]
                mat_block = Sa @ Pa @ Sa
                ovlp_block = Sa
            elif mode == "gross" or mode in ["lowdin", "meta-lowdin"]:
                mat_block = P_mol[p0:p1, p0:p1]
                ovlp_block = np.eye(p1 - p0)
            elif mode == "sym":
                Pa, Sa = P_mol[p0:p1, p0:p1], S_mol[p0:p1, p0:p1]
                mat_block = (Pa @ Sa + Sa @ Pa) / 2
                ovlp_block = np.eye(p1 - p0)
            elif mode == "sps":
                mat_block = (S_mol @ P_mol @ S_mol)[p0:p1, p0:p1]
                ovlp_block = np.eye(p1 - p0)
            elif mode == "spsa":
                mat_block = (S_mol @ P_mol @ S_mol)[p0:p1, p0:p1]
                ovlp_block = S_mol[p0:p1, p0:p1]
            else:
                mat_block = P_mol[p0:p1, p0:p1]
                ovlp_block = S_mol[p0:p1, p0:p1]
            w, c, l_map = spherical_average(mol, ia, mat_block, ovlp_block)

        final_idx = []
        unique_l = sorted(target_l_counts.keys())
        for l in unique_l:
            target_n_shells = target_l_counts[l]
            l_idx = np.where(l_map == l)[0]
            if len(l_idx) == 0: continue
            w_l = w[l_idx]
            # Identify shells by unique eigenvalues (degenerate components share same eigenvalue)
            _, shell_start_indices = np.unique(np.round(w_l, 10), return_index=True)
            shell_start_indices = np.sort(shell_start_indices)
            for s_idx in range(min(target_n_shells, len(shell_start_indices))):
                start = shell_start_indices[s_idx]
                end = shell_start_indices[s_idx+1] if s_idx+1 < len(shell_start_indices) else len(l_idx)
                final_idx.extend(l_idx[start:end])

        w_keep = w[final_idx]
        c_keep = c[:, final_idx]
        veps_block[p0:p1, col_idx: col_idx + n_target] = c_keep
        vaps_diag.extend(w_keep)
        col_idx += n_target

    print(f"\nFinal EffAO Eigenvalues for molecule (mode={mode}):")
    print(np.array(vaps_diag))
    return np.array(vaps_diag), veps_block, pmol

def _do_iao(mol, coeffs, pmol=None, A_basis=None):
    from pyscf.lo import iao as pyscf_iao
    from pyscf.lo import orth
    S1 = mol.intor('int1e_ovlp')
    if A_basis is not None:
        A_tilde = A_basis
        S2 = A_tilde.T @ S1 @ A_tilde
        S12 = S1 @ A_tilde
        C_min = scipy.linalg.solve(S2, S12.T @ coeffs, assume_a='pos')
        C_proj = orth.vec_lowdin(A_tilde @ C_min, S1)
        P_occ_A = coeffs @ (coeffs.T @ S12)
        P_proj_A = C_proj @ (C_proj.T @ S12)
        P_occ_P_proj_A = coeffs @ (coeffs.T @ (S1 @ P_proj_A))
        IAO_nonorth = A_tilde + 2 * P_occ_P_proj_A - P_occ_A - P_proj_A
        return orth.vec_lowdin(IAO_nonorth, S1)
    else:
        C_iao_nonorth = pyscf_iao.iao(mol, coeffs, minao=pmol.basis)
        return orth.vec_lowdin(C_iao_nonorth, S1)

def iao(mol, coeffs, source_basis='minao', pol_basis=None):
    pmol = reference_mol(mol, polarized=(pol_basis is not None), pol_basis=pol_basis, source_basis=source_basis)
    return _do_iao(mol, coeffs, pmol=pmol), pmol

def fpiao(mol, coeffs, x=1.0, source_basis='minao', pol_basis='ano'):
    pmol = reference_mol(mol, polarized=True, pol_basis=pol_basis, source_basis=source_basis, x=x)
    return _do_iao(mol, coeffs, pmol=pmol), pmol

def autosad(mol, mf, free_atom=True, mode=None, polarized=False):
    def do_autosad(mol, mf, C_occ, free_atom, mode, polarized):
        w, effaos, pmol = get_effaos(mol, mf, free_atom=free_atom, mode=mode, polarized=polarized)
        return _do_iao(mol, C_occ, A_basis=effaos)
    if "U" in mf.__class__.__name__:
        ca, cb = mf.mo_coeff; return (do_autosad(mol, mf, ca[:, :mf.nelec[0]], free_atom, mode, polarized), 
                                      do_autosad(mol, mf, cb[:, :mf.nelec[1]], free_atom, mode, polarized)), reference_mol(mol)
    return do_autosad(mol, mf, mf.mo_coeff[:, :mol.nelectron // 2], free_atom, mode, polarized), reference_mol(mol)

def dfpiao(mol, coeffs, x=0.5, source_basis='minao', pol_basis='ano'):
    res_iao = iao(mol, coeffs, source_basis=source_basis)
    res_fpiao = fpiao(mol, coeffs, x=1.0, source_basis=source_basis, pol_basis=pol_basis)
    return res_iao, res_fpiao
