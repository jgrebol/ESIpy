import os
import numpy as np
import scipy.linalg
from pyscf import gto, scf
from pyscf.data import elements
from pyscf.lo import orth
from pyscf.lo.nao import _sph_average_mat, _cart_average_mat

def load_iao_dat_basis(file_path, symbol):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    basis = []
    found_element = False
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('ELEMENT'):
            parts = line.split()
            if len(parts) >= 3 and parts[2].upper() == symbol.upper():
                found_element = True
                i += 1
                while i < len(lines) and 'NSHELL' not in lines[i]: i += 1
                if i < len(lines):
                    num_shells = int(lines[i].strip().split()[1])
                    i += 1
                    for _ in range(num_shells):
                        while i < len(lines) and 'SHELL' not in lines[i]: i += 1
                        if i >= len(lines): break
                        shell_parts = lines[i].strip().split()
                        l, nprim = int(shell_parts[1]), int(shell_parts[2])
                        i += 1
                        shell = [l]
                        for _ in range(nprim):
                            prim_parts = lines[i].strip().split()
                            shell.append([float(prim_parts[0]), float(prim_parts[1])])
                            i += 1
                        basis.append(shell)
                break
        i += 1
    if not found_element: raise ValueError(f'Element {symbol} not found')
    return basis

def _load_basis_wrapper(name, sym):
    if os.path.exists(name):
        return load_iao_dat_basis(name, sym)
    from pyscf import gto
    return gto.basis.load(name, sym)

def get_union_occ(mol, mf):
    """Returns a basis for the union of alpha and beta occupied spaces, or the occupied space for RHF."""
    if hasattr(mf, 'mo_coeff') and hasattr(mf, 'mo_occ'):
        if np.shape(mf.mo_occ) == 1:
            return mf.mo_coeff[:, mf.mo_occ > 1e-10]
        else:
            a, b = mf.mo_coeff
            aocc, bocc = mf.mo_occ
            return a[:, aocc > 1e-10] + b[:, bocc > 1e-10]
            # UHF: Return a basis for the union of alpha and beta occupied spaces
            # Using the density matrix approach but more robustly
            pass

    S = mol.intor('int1e_ovlp')
    if hasattr(mf, 'make_rdm1'):
        dm = mf.make_rdm1()
    else:
        # Assume mf is mo_coeff or density matrix
        if isinstance(mf, (list, tuple)):
            # UHF coefficients (occupied only)
            dm = mf[0] @ mf[0].T + mf[1] @ mf[1].T
        else:
            dm = mf
    
    if dm.ndim == 3:
        dm_total = dm[0] + dm[1]
    else:
        dm_total = dm
        
    try:
        e, c = scipy.linalg.eigh(dm_total, S)
    except scipy.linalg.LinAlgError:
        # Fallback for near-singular overlap
        e_s, u_s = scipy.linalg.eigh(S)
        mask = e_s > 1e-12
        s_inv_half = u_s[:, mask] @ np.diag(1.0 / np.sqrt(e_s[mask])) @ u_s[:, mask].T
        e, v = np.linalg.eigh(s_inv_half.T @ dm_total @ s_inv_half)
        c = s_inv_half @ v

    # Occupations are between 0 and 2. We take orbitals with significant occupation.
    return c[:, e > 1e-10]

def dump_matrix(path, a):
    with open(path, 'w') as f:
        f.write(f"{a.shape[0]} {a.shape[1]}\n")
        for row in a:
            f.write(" ".join([f"{x:24.16E}" for x in row]) + "\n")

def spherical_average(mol, ia, mat, overlap):
    """
    Spherical averaging following the logic in iaoeffaos.py
    """
    ao_loc = mol.ao_loc_nr()
    bas_ang = mol._bas[:, gto.ANG_OF]

    all_w, l_map, shell_map = [], [], []
    all_c = np.zeros((mat.shape[0], mat.shape[0]))
    current_col = 0

    b0, b1, p0_atom, p1_atom = mol.aoslice_by_atom()[ia]
    if b1 <= b0: return np.array(all_w), all_c, np.array(l_map), np.array(shell_map)

    l_max = bas_ang[b0:b1].max()
    for l in range(l_max + 1):
        ib_l = np.where(bas_ang[b0:b1] == l)[0]
        if len(ib_l) == 0: continue

        idx = np.hstack([np.arange(ao_loc[b0+ib] - p0_atom, ao_loc[b0+ib+1] - p0_atom) for ib in ib_l])
        degen = (l + 1) * (l + 2) // 2 if mol.cart else l * 2 + 1
        n_shells = len(idx) // degen
        idx_reshaped = idx.reshape(-1, degen)

        p_frag = np.zeros((n_shells, n_shells))
        s_frag = np.zeros((n_shells, n_shells))
        for i in range(n_shells):
            for j in range(n_shells):
                p_frag[i, j] = np.trace(mat[np.ix_(idx_reshaped[i], idx_reshaped[j])]) / degen
                s_frag[i, j] = np.trace(overlap[np.ix_(idx_reshaped[i], idx_reshaped[j])]) / degen

        try:
            e, v = scipy.linalg.eigh(p_frag, s_frag)
        except:
            e, v = scipy.linalg.eigh(p_frag)

        sort_idx = np.argsort(e)[::-1]
        e, v = e[sort_idx], v[:, sort_idx]

        for iw in range(n_shells):
            val = e[iw]
            for m in range(degen):
                v_full = np.zeros(mat.shape[0])
                for i_shell in range(n_shells):
                    v_full[idx_reshaped[i_shell, m]] = v[i_shell, iw]
                norm = np.sqrt(v_full.T @ overlap @ v_full)
                if norm > 1e-14: v_full /= norm
                all_w.append(val)
                l_map.append(l)
                shell_map.append(iw)
                all_c[:, current_col] = v_full
                current_col += 1

    return np.array(all_w), all_c, np.array(l_map), np.array(shell_map)

def get_num_minbas_per_l(sym, polarized=False):
    z = elements.charge(sym)
    if z <= 2:    d = {0: 1}  # 1s
    elif z <= 10:  d = {0: 2, 1: 1}  # 1s, 2s, 2p
    elif z <= 18:  d = {0: 3, 1: 2}  # 3s, 3p
    elif z <= 36:  d = {0: 4, 1: 3, 2: 1}  # 4s, 4p, 3d
    elif z <= 54:  d = {0: 5, 1: 4, 2: 2}  # 5s, 5p, 4d
    elif z <= 86:  d = {0: 6, 1: 5, 2: 3, 3: 1}
    elif z <= 118: d = {0: 7, 1: 6, 2: 4, 3: 2}
    else: raise NotImplementedError(f"Minimal basis not defined for: {sym}")
    if polarized:
        if z <= 2: d[1] = 1 # 2p
        elif z <= 10: d[2] = 1 # 3d
        elif z <= 18: d[2] = 1 # 3d
        elif z <= 36: d[3] = 1 # 4f
        elif z <= 54: d[3] = 1 # 4f
    return d

def get_num_minbas_ao(mol, ia, polarized=False, heavy_only=True):
    sym = mol.atom_pure_symbol(ia)
    if polarized and heavy_only and sym == 'H':
        target_l = get_num_minbas_per_l(sym, polarized=False)
    else:
        target_l = get_num_minbas_per_l(sym, polarized=polarized)
    total = 0
    for l, count in target_l.items():
        # Find the degeneracy for this l in the molecule
        degen = 2*l + 1
        for ib in range(mol.nbas):
            if mol.bas_atom(ib) == ia and mol.bas_angular(ib) == l:
                # De-reference the number of functions in the shell
                degen = (mol.ao_loc_nr()[ib+1] - mol.ao_loc_nr()[ib]) // mol.bas_nctr(ib)
                break
        total += count * degen
    return total

def get_reference_basis_dict(mol, source_basis='minao', pol_basis=None, x=1.0, heavy_only=False, full_basis=False):
    def get_minimal_part(sym, source_basis_name):
        if full_basis: return _load_basis_wrapper(source_basis_name, sym), {}
        basis_source = _load_basis_wrapper(source_basis_name, sym)
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
            basis_pol = _load_basis_wrapper(pol_basis_name, sym)
        else: return []
        target_pol_l = get_num_minbas_per_l(sym, polarized=True)
        new_pol_atom, l_counts = [], {}
        for shell in basis_pol:
            l = shell[0]
            if l in target_pol_l and l not in target_l:
                if l_counts.get(l, 0) < target_pol_l[l]:
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
            pol_basis_list = get_polarization_part(sym, pol_basis, target_l, x=x) if not full_basis else []
            ref_basis[sym] = min_basis + pol_basis_list
    return ref_basis

def reference_mol(mol, polarized=False, pol_basis=None, source_basis='minao', x=1.0, heavy_only=False, full_basis=False):
    from pyscf.lo import iao as pyscf_iao
    if hasattr(mol, 'pyscf_mol'): mol = mol.pyscf_mol
    if polarized and pol_basis is None: pol_basis = 'ano'
    elif not polarized: pol_basis = None
    ref_basis = get_reference_basis_dict(mol, source_basis=source_basis, pol_basis=pol_basis, x=x, heavy_only=heavy_only, full_basis=full_basis)
    print("Using the reference basis:", ref_basis)
    if heavy_only:
        for sym in ref_basis:
            if sym == 'H':
                new_shells = []
                for shell in ref_basis[sym]:
                    if shell[0] == 0: new_shells.append(shell)
                ref_basis[sym] = new_shells
    return pyscf_iao.reference_mol(mol, minao=ref_basis)

atom_spins = {'H':1,'He':0,'Li':1,'Be':0,'B':1,'C':2,'N':3,'O':2,'F':1,'Ne':0,'Na':1,'Mg':0,'Al':1,'Si':2,'P':3,'S':2,'Cl':1,'Ar':0}

def get_effaos(mol, mf, free_atom=True, mode='net', polarized=False, heavy_only=False, full_basis=False, x=1.0):
    pmol = reference_mol(mol, polarized=polarized, heavy_only=heavy_only, full_basis=full_basis, x=x)
    minbas_total = pmol.nao
    veps_block = np.zeros((mol.nao, minbas_total))
    vaps_diag = []
    S_mol = mol.intor("int1e_ovlp")
    
    if hasattr(mf, 'make_rdm1'):
        dm = mf.make_rdm1()
        if dm.ndim == 3: P_mol = dm[0] + dm[1]
        else: P_mol = dm
    else:
        P_mol = mf

    T_orth = None  # TRACK THE ORTHOGONALIZATION MATRIX

    if not free_atom:
        print(f"| Eff-AO mode: {mode}")
        if mode in ["lowdin", "meta-lowdin", "ml"]:
            if mode == "lowdin":
                T_orth = orth.lowdin(S_mol)
            else:
                from pyscf.lo.orth import orth_ao
                # mf is explicitly required here for occupations
                T_orth = orth_ao(mf, 'meta-lowdin')

            T_inv = T_orth.T @ S_mol
            P_mol = T_inv @ P_mol @ T_inv.T
            S_mol = np.eye(mol.nao)
        elif mode == "gross":
            PS = P_mol @ S_mol
            P_mol = (PS + PS.T) * 0.5
            S_mol = np.eye(mol.nao)

    print(f"| Total population (trace of PS): {np.trace(P_mol @ S_mol):.10f}")
    total_eff_pop = 0
    total_kept_pop = 0

    aoslices = mol.aoslice_by_atom()
    col_idx = 0
    
    for ia in range(mol.natm):
        sym = mol.atom_pure_symbol(ia)
        p0, p1 = aoslices[ia, 2], aoslices[ia, 3]
        target_l_counts = get_num_minbas_per_l(sym, polarized=polarized)
        n_target = get_num_minbas_ao(mol, ia, polarized=polarized, heavy_only=heavy_only)

        if free_atom:
            atom_mol = gto.M(atom=f"{sym} 0 0 0", basis=mol.basis, spin=atom_spins.get(sym, 0), charge=0, cart=mol.cart)
            if hasattr(mf, 'xc'):
                atom_mf = scf.RKS(atom_mol) if atom_mol.spin == 0 else scf.UKS(atom_mol)
                atom_mf.xc = mf.xc
            else:
                atom_mf = scf.RHF(atom_mol) if atom_mol.spin == 0 else scf.UHF(atom_mol)
            atom_mf.verbose = 0
            atom_mf.kernel()
            S_at = atom_mol.intor('int1e_ovlp')
            P_at = atom_mf.make_rdm1()
            if P_at.ndim == 3: P_at = P_at[0] + P_at[1]
            mat_block = S_at @ P_at @ S_at
            ovlp_block = S_at
            w, c, l_map, shell_map = spherical_average(atom_mol, 0, mat_block, ovlp_block)
        else:
            if mode == "net":
                Sa, Pa = S_mol[p0:p1, p0:p1], P_mol[p0:p1, p0:p1]
                mat_block = Sa @ Pa @ Sa
                ovlp_block = Sa
            elif mode == "gross" or mode in ["lowdin", "meta-lowdin", "ml"]:
                mat_block = P_mol[p0:p1, p0:p1]
                ovlp_block = np.eye(p1 - p0)
            elif mode in ["symmetric", "sym"]:
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
            w, c, l_map, shell_map = spherical_average(mol, ia, mat_block, ovlp_block)

        total_eff_pop += np.sum(w)

        final_idx = []
        unique_l = sorted(target_l_counts.keys())
        for l in unique_l:
            target_n_shells = target_l_counts[l]
            # Grab strictly the required number of top-occupied shells
            # Using unique eigenvalues to identify shells as in iaoeffaos.py logic
            l_idx = np.where(l_map == l)[0]
            if len(l_idx) == 0: continue
            
            # Group by unique eigenvalues (shells)
            unique_w, shell_start = np.unique(np.round(w[l_idx], 8), return_index=True)
            # Sort shell_start to correspond to descending eigenvalues (already sorted)
            # Actually unique returns sorted ascending, so we need to reverse
            shell_start = np.sort(shell_start)[::-1]
            
            # The selection logic in iaoeffaos.py:
            # shell_start = np.sort(shell_start)
            # for s_idx in range(min(target_n, len(shell_start))):
            #    ...
            # Wait, iaoeffaos.py does np.argsort(e)[::-1] in spherical_average,
            # so shells are already ordered by descending eigenvalues.
            # My spherical_average also does sort_idx = np.argsort(e)[::-1].
            
            # Let's just use shell_map which I've correctly assigned.
            idx_keep = np.where((l_map == l) & (shell_map < target_n_shells))[0]
            final_idx.extend(idx_keep)

        final_idx = np.sort(final_idx)

        w_keep = w[final_idx]
        total_kept_pop += np.sum(w_keep)
        c_keep = c[:, final_idx]
        veps_block[p0:p1, col_idx: col_idx + n_target] = c_keep
        vaps_diag.extend(w_keep)
        col_idx += n_target

    # TRANSFORM BACK TO NON-ORTHOGONAL AO BASIS
    if T_orth is not None:
        veps_block = T_orth @ veps_block

    print(f"| Sum of all Eff-AO eigenvalues: {total_eff_pop:.10f}")
    print(f"| Sum of kept Eff-AO eigenvalues: {total_kept_pop:.10f}")

    print(f"\nFinal EffAO Occupations for molecule (mode={mode}):")
    print(np.array(vaps_diag))
    return np.array(vaps_diag), veps_block, pmol

def _do_iao(mol, coeffs, pmol=None, A_basis=None, heavy_only=False):
    from pyscf.lo import iao as pyscf_iao
    from pyscf.lo import orth
    S1 = mol.intor('int1e_ovlp')
    if A_basis is not None:
        A_tilde = A_basis
        S2 = A_tilde.T @ S1 @ A_tilde
        S12 = S1 @ A_tilde
        import os
        prefix = os.environ.get("IAO_DUMP_PREFIX")
        if prefix:
            dump_matrix(prefix + "_s2.dat", S2)
            dump_matrix(prefix + "_s12.dat", S12)
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

def iao(mol, coeffs, source_basis='minao', heavy_only=False, full_basis=False):
    if not isinstance(coeffs, np.ndarray):
        coeffs = get_union_occ(mol, coeffs)
    pmol = reference_mol(mol, polarized=False, source_basis=source_basis, heavy_only=heavy_only, full_basis=full_basis)
    return _do_iao(mol, coeffs, pmol=pmol), pmol

def fpiao(mol, coeffs, x=1.0, source_basis='minao', pol_basis='ano', heavy_only=True, full_basis=False):
    if not isinstance(coeffs, np.ndarray):
        coeffs = get_union_occ(mol, coeffs)
    pmol = reference_mol(mol, polarized=True, pol_basis=pol_basis, source_basis=source_basis, x=x, heavy_only=heavy_only, full_basis=full_basis)
    return _do_iao(mol, coeffs, pmol=pmol), pmol

def autosad(mol, mf, polarized=False, heavy_only=False, full_basis=False, x=1.0):
    c_occ = get_union_occ(mol, mf)
    w, A, pmol = get_effaos(mol, mf, free_atom=True, polarized=polarized, heavy_only=heavy_only, full_basis=full_basis, x=x)
    return _do_iao(mol, c_occ, A_basis=A, heavy_only=heavy_only), pmol

def effao(mol, mf, mode='net', polarized=False, heavy_only=False, full_basis=False, x=1.0):
    w, A, pmol = get_effaos(mol, mf, free_atom=False, mode=mode, polarized=polarized, heavy_only=heavy_only, full_basis=full_basis, x=x)
    c_occ = get_union_occ(mol, mf)
    return _do_iao(mol, c_occ, A_basis=A, heavy_only=heavy_only), pmol

def wiao(mol, mf, heavy_only=False, full_basis=False):
    from pyscf.lo.orth import orth_ao
    
    # 1. Get meta-Lowdin transformation matrix (whole basis)
    S1 = mol.intor('int1e_ovlp')
    T = orth_ao(mf, 'iao')
    
    # 2. Get density matrix in meta-Lowdin basis: P_ML = T.T @ S1 @ P_AO @ S1 @ T
    if hasattr(mf, 'make_rdm1'):
        dm = mf.make_rdm1()
        if dm.ndim == 3: P_AO = dm[0] + dm[1]
        else: P_AO = dm
    else:
        P_AO = mf
    
    T_inv = T.T @ S1
    P_ML = T_inv @ P_AO @ T_inv.T
    
    # 3. Compute whole-basis bond order matrix B_whole = P_ML @ P_ML
    B_whole = P_ML @ P_ML
    
    # 4. Aggregate B_whole into atomic matrix B_atom (NxN atoms)
    natm = mol.natm
    B_atom = np.zeros((natm, natm))
    ao_loc = mol.ao_loc_nr()
    
    for i in range(natm):
        for j in range(natm):
            # Check for Hydrogen atoms and set to zero if either is H
            if mol.atom_pure_symbol(i) == 'H' or mol.atom_pure_symbol(j) == 'H':
                B_atom[i, j] = 0.0
            else:
                # Sum all orbital matrix elements for atoms i and j
                block = B_whole[ao_loc[i]:ao_loc[i+1], ao_loc[j]:ao_loc[j+1]]
                B_atom[i, j] = np.sum(block)
    
    # 5. Compute atomic Pearson Correlation Coefficient PCC_atom
    diag_B = np.diag(B_atom)
    diag_B = np.where(diag_B > 1e-6, diag_B, 1.0)
    denom = np.sqrt(np.outer(diag_B, diag_B))
    PCC_atom = B_atom / denom

    # Zero out Hydrogen mixing and enforce perfect 1.0 on diagonals
    for i in range(natm):
        if mol.atom_pure_symbol(i) == 'H':
            PCC_atom[i, :] = 0.0
            PCC_atom[:, i] = 0.0
            PCC_atom[i, i] = 1.0
        else:
            PCC_atom[i, i] = 1.0
            
    # 6. Get minimal basis orbitals A_tilde and pmol
    _, A_tilde, pmol = get_effaos(mol, mf, free_atom=False, mode='meta-lowdin', 
                                  polarized=False, heavy_only=heavy_only, 
                                  full_basis=full_basis)
    
    # 7. Broadcast PCC_atom into minimal basis blocks of PCC_min
    nmin = pmol.nao
    PCC_min = np.zeros((nmin, nmin))
    loc_min = pmol.ao_loc_nr()
    
    for i in range(natm):
        for j in range(natm):
            PCC_min[loc_min[i]:loc_min[i+1], loc_min[j]:loc_min[j+1]] = PCC_atom[i, j]
            
    # Regularization to ensure non-singularity (especially within atomic blocks)
    PCC_min = 0.999 * PCC_min + 0.001 * np.eye(nmin)
    
    # 8. Compute weighted reference basis A_wiao = A_tilde @ PCC_min
    A_wiao = A_tilde @ PCC_min
    c_occ = get_union_occ(mol, mf)
    
    return _do_iao(mol, c_occ, A_basis=A_wiao), pmol

def dfpiao(mol, coeffs, x=0.5, source_basis='minao', pol_basis='ano', heavy_only=True, full_basis=False):
    if not isinstance(coeffs, np.ndarray):
        coeffs = get_union_occ(mol, coeffs)
    return fpiao(mol, coeffs, x=x, source_basis=source_basis, pol_basis=pol_basis, heavy_only=heavy_only, full_basis=full_basis)
