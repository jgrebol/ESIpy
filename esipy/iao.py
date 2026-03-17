import numpy as np
import scipy.linalg
from pyscf import gto, scf
from pyscf.data import elements
from pyscf.lo import orth

def spherical_average(mol, ia, mat, overlap):
    """
    Performs spherical averaging of a property matrix for atom 'ia'.
    Uses PySCF's native _prenao_sub logic for AO slicing and averaging.
    Returns eigenvalues, eigenvectors (in full AO space), and an L-map.
    """
    ao_loc = mol.ao_loc_nr()
    bas_ang = mol._bas[:, gto.ANG_OF]

    all_w = []
    all_c = np.zeros((mat.shape[0], mat.shape[0]))
    l_map = []
    current_col = 0

    # Get basis slice for the target atom
    b0, b1, p0_atom, p1_atom = mol.aoslice_by_atom()[ia]
    if b1 <= b0:
        return np.array(all_w), all_c, np.array(l_map)

    l_max = bas_ang[b0:b1].max()

    for l in range(l_max + 1):
        # Find all shells on this atom with angular momentum l
        ib_l = np.where(bas_ang[b0:b1] == l)[0]
        if len(ib_l) == 0:
            continue

        # Extract all AO indices for this L
        idx = []
        for ib in ib_l:
            idx.append(np.arange(ao_loc[b0+ib] - p0_atom, ao_loc[b0+ib+1] - p0_atom))
        idx = np.hstack(idx)

        # Determine degeneracy (Spherical vs Cartesian)
        if mol.cart:
            degen = (l + 1) * (l + 2) // 2
        else:
            degen = l * 2 + 1

        n_shells = len(idx) // degen
        idx_reshaped = idx.reshape(-1, degen)

        # Average the matrix blocks over the degenerate components
        p_frag = np.zeros((n_shells, n_shells))
        s_frag = np.zeros((n_shells, n_shells))
        for i in range(n_shells):
            for j in range(n_shells):
                p_frag[i, j] = np.trace(mat[np.ix_(idx_reshaped[i], idx_reshaped[j])]) / degen
                s_frag[i, j] = np.trace(overlap[np.ix_(idx_reshaped[i], idx_reshaped[j])]) / degen

        # Solve Generalized Eigenvalue Problem directly as PySCF does
        try:
            e, v = scipy.linalg.eigh(p_frag, s_frag)
        except scipy.linalg.LinAlgError:
            # Fallback if overlap is ill-conditioned
            e, v = scipy.linalg.eigh(p_frag)

        # Sort descending by occupation
        sort_idx = np.argsort(e)[::-1]
        e = e[sort_idx]
        v = v[:, sort_idx]

        print(f"L={l} Shell Occupations (Averaged): {e}")

        # Map back to full AO space and duplicate for each degenerate component
        for iw in range(n_shells):
            val = e[iw]
            for m in range(degen):
                v_full = np.zeros(mat.shape[0])

                # Scatter the reduced eigenvector back into the full AO index map
                for i_shell in range(n_shells):
                    v_full[idx_reshaped[i_shell, m]] = v[i_shell, iw]

                # Normalize the eigenvector in the full overlap metric
                norm = np.sqrt(v_full.T @ overlap @ v_full)
                if norm > 1e-14:
                    v_full /= norm

                all_w.append(val)
                l_map.append(l)  # <--- Tracking L corresponding to this eigenvalue
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
                    # Build a new shell preserving all primitives and contraction coefficients
                    new_shell = [l]
                    for prim in shell[1:]:
                        scaled_exp = prim[0] * x
                        contraction_coeff = prim[1]
                        new_shell.append([scaled_exp, contraction_coeff])
                    new_pol_atom.append(new_shell)
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
        dm = mf.make_rdm1(ao_repr=True)
        P_mol = dm[0] + dm[1] if dm.ndim == 3 else dm
    else:
        P_mol = mf.make_rdm1(ao_repr=True)

    if not free_atom:
        if mode in ["lowdin", "meta-lowdin"]:
            if mode == "lowdin":
                T = orth.lowdin(S_mol)
            else:
                from pyscf.lo.orth import orth_ao
                T = orth_ao(mf, 'meta_lowdin')

            T_inv = T.T @ S_mol # T^{T} S T = I  --> T^{T} S = T^{-1} 
            P_mol = T_inv @ P_mol @ T_inv.T
            S_mol = np.eye(mol.nao)

        elif mode == "gross":
            P_mol = (P_mol @ S_mol + S_mol @ P_mol) / 2 # ((PS + SP)^A)/2
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
            mat_block = S_at @ P_at @ S_at # S P S c = S c L
            ovlp_block = S_at
            w, c, l_map = spherical_average(mol_at, 0, mat_block, ovlp_block)
        else:
            if mode == "net":
                Sa, Pa = S_mol[p0:p1, p0:p1], P_mol[p0:p1, p0:p1]
                mat_block = Sa @ Pa @ Sa # S^A P^A S^A
                ovlp_block = Sa # S^A
            elif mode == "gross" or mode in ["lowdin", "meta-lowdin"]:
                mat_block = P_mol[p0:p1, p0:p1] # ((PS + SP)^A)/2
                ovlp_block = np.eye(p1 - p0) # I
            elif mode == "sym":
                Pa, Sa = P_mol[p0:p1, p0:p1], S_mol[p0:p1, p0:p1]
                mat_block = (Pa @ Sa + Sa @ Pa) / 2 # (P^A S^A + S^A P^A)/2
                ovlp_block = np.eye(p1 - p0) # I
            elif mode == "sps":
                mat_block = (S_mol @ P_mol @ S_mol)[p0:p1, p0:p1] # (SPS)^A
                ovlp_block = np.eye(p1 - p0) # I
            elif mode == "spsa":
                mat_block = (S_mol @ P_mol @ S_mol)[p0:p1, p0:p1] # (SPS)^A
                ovlp_block = S_mol[p0:p1, p0:p1] # S^A
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

            # Truncation: Keep only the target_n_shells (e.g., top 1s, 2s, 2p)
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

