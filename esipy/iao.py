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


import os
import numpy as np
import scipy.linalg
from pyscf import gto, scf
from pyscf.data import elements
from pyscf.lo import orth
from pyscf.lo.nao import _sph_average_mat, _cart_average_mat

def spherical_average(mol, ia, mat, overlap):
    """
    Exact clone of PySCF's _prenao_sub logic applied to a single atom block,
    augmented to keep track of angular momentum (l_map) and shell rank (shell_map)
    for robust minimal basis extraction.
    """
    ao_loc = mol.ao_loc_nr()
    aoslices = mol.aoslice_by_atom()
    b0, b1, p0, p1 = aoslices[ia]
    nao_atom = p1 - p0

    occ = np.zeros(nao_atom)
    cao = np.zeros((nao_atom, nao_atom), dtype=overlap.dtype)
    l_map = np.full(nao_atom, -1, dtype=int)
    shell_map = np.full(nao_atom, -1, dtype=int)

    # Trace conservation check
    if np.allclose(overlap, np.eye(overlap.shape[0])):
        trace_before = np.trace(mat)
    else:
        try:
            # Trace of generalized eigenvalue problem is trace(S^-1 P)
            trace_before = np.trace(scipy.linalg.solve(overlap, mat, assume_a='pos'))
        except:
            trace_before = np.trace(np.linalg.pinv(overlap) @ mat)

    bas_ang = mol._bas[:, gto.ANG_OF]
    if b1 <= b0:
        return occ, cao, l_map, shell_map

    l_max = bas_ang[b0:b1].max()

    for l in range(l_max + 1):
        idx = []
        for ib in np.where(bas_ang[b0:b1] == l)[0]:
            # Make indices relative to the atom block
            idx.append(np.arange(ao_loc[b0+ib] - p0, ao_loc[b0+ib+1] - p0))
        if len(idx) == 0:
            continue
        idx = np.hstack(idx)

        # Spherical or Cartesian averaging
        if mol.cart:
            degen = (l + 1) * (l + 2) // 2
            p_frag = _cart_average_mat(mat, l, idx)
            s_frag = _cart_average_mat(overlap, l, idx)
        else:
            degen = 2 * l + 1
            p_frag = _sph_average_mat(mat, l, idx)
            s_frag = _sph_average_mat(overlap, l, idx)

        # Solve Generalized Eigenvalue Problem
        try:
            e, v = scipy.linalg.eigh(p_frag, s_frag)
        except scipy.linalg.LinAlgError:
            # Fallback for near-singular overlap
            e_s, u_s = scipy.linalg.eigh(s_frag)
            mask = e_s > 1e-12
            s_inv_half = u_s[:, mask] @ np.diag(1.0 / np.sqrt(e_s[mask])) @ u_s[:, mask].T
            e, v_prime = scipy.linalg.eigh(s_inv_half @ p_frag @ s_inv_half)
            v = s_inv_half @ v_prime

        # Sort descending (This makes shell 0 the highest occ, shell 1 the next, etc.)
        e = e[::-1]
        v = v[:, ::-1]

        # Map back to full atom block
        idx = idx.reshape(-1, degen)
        for k in range(degen):
            ilst = idx[:, k]
            occ[ilst] = e
            l_map[ilst] = l
            # Directly map the shell rank from the sorted eigenvalue array
            shell_map[ilst] = np.arange(len(e))
            for i, i0 in enumerate(ilst):
                cao[i0, ilst] = v[i]

    trace_after = np.sum(occ)
    print(f"    Atom {ia:3d} ({mol.atom_symbol(ia):2s}): Trace before average = {trace_before:10.6f}, Trace after = {trace_after:10.6f}, Diff = {trace_before-trace_after:10.2e}")

    return occ, cao, l_map, shell_map

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
        if z <= 2: d[1] = 1
        elif z <= 10: d[2] = 1
        elif z <= 18: d[2] = 1
        elif z <= 36: d[3] = 1
        elif z <= 54: d[3] = 1
    return d

def get_num_minbas_ao(mol, ia, polarized=False):
    """
    Calculates the total number of minimal basis functions for a given atom,
    accounting for whether the molecule is using spherical or Cartesian basis.
    """
    sym = mol.atom_pure_symbol(ia)
    l_counts = get_num_minbas_per_l(sym, polarized=polarized)

    n_target = 0
    for l, count in l_counts.items():
        if mol.cart:
            degen = (l + 1) * (l + 2) // 2
        else:
            degen = 2 * l + 1
        n_target += count * degen

    return n_target

def _load_basis_wrapper(name, sym):
    if os.path.exists(name):
        return load_iao_dat_basis(name, sym)
    from pyscf import gto
    return gto.basis.load(name, sym)

def get_reference_basis_dict(mol, source_basis='minao', pol_basis=None, x=1.0, heavy_only=False, full_basis=False):
    def get_minimal_part(sym, source_basis_name):
        if full_basis:
            if source_basis_name == 'working': return mol._basis[sym], {}
            return _load_basis_wrapper(source_basis_name, sym), {}
        if source_basis_name == 'working':
            basis_source = mol._basis[sym]
        else:
            basis_source = _load_basis_wrapper(source_basis_name, sym)
        target_l = get_num_minbas_per_l(sym, polarized=False)
        new_basis_atom, l_counts = [], {}
        for shell in basis_source:
            l = shell[0]
            # In PySCF format [l, [exp, c1, c2...], ...], the number of functions is len(shell[1]) - 1
            n_func = len(shell[1]) - 1
            count = l_counts.get(l, 0)
            if count < target_l.get(l, 0):
                # If this shell provides more functions than needed, we must truncate the contraction coefficients
                needed = target_l[l] - count
                if n_func > needed:
                    # Truncate shell to only take 'needed' coefficients
                    new_shell = [l]
                    for prim in shell[1:]:
                        new_shell.append(prim[:needed+1])
                    new_basis_atom.append(new_shell)
                    l_counts[l] = count + needed
                else:
                    new_basis_atom.append(shell)
                    l_counts[l] = count + n_func
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
            # polarization part only adds l NOT in target_l (the minimal part)
            if l in target_pol_l and l not in target_l:
                n_func = len(shell[1]) - 1
                count = l_counts.get(l, 0)
                if count < target_pol_l[l]:
                    needed = target_pol_l[l] - count
                    new_shell = [l]
                    if n_func > needed:
                        for prim in shell[1:]:
                            new_shell.append([prim[0] * x] + prim[1:needed+1])
                        l_counts[l] = count + needed
                    else:
                        for prim in shell[1:]:
                            new_shell.append([prim[0] * x] + prim[1:])
                        l_counts[l] = count + n_func
                    new_pol_atom.append(new_shell)
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
    ref_basis = get_reference_basis_dict(mol, source_basis=source_basis, pol_basis=pol_basis, x=x)
    print("Using the reference basis:", ref_basis)
    if heavy_only:
        for sym in ref_basis:
            if sym == 'H':
                new_shells = []
                for shell in ref_basis[sym]:
                    if shell[0] == 0: new_shells.append(shell)
                ref_basis[sym] = new_shells
    return pyscf_iao.reference_mol(mol, minao=ref_basis)

def get_effaos(mol, mf, free_atom=True, mode=None, polarized=False, heavy_only=False, full_basis=False):
    pmol = reference_mol(mol, polarized=polarized, heavy_only=heavy_only, full_basis=full_basis)
    minbas_total = pmol.nao
    veps_block = np.zeros((mol.nao, minbas_total))
    vaps_diag = []
    S_mol = mol.intor("int1e_ovlp")
    if "U" in mf.__class__.__name__:
        dm = mf.make_rdm1(ao_repr=True); P_mol = dm[0] + dm[1] if dm.ndim == 3 else dm
    else:
        P_mol = mf.make_rdm1(ao_repr=True)

    T_orth = None  # TRACK THE ORTHOGONALIZATION MATRIX

    if not free_atom:
        print(f"| Eff-AO mode: {mode}")
        if mode in ["lowdin", "meta-lowdin"]:
            if mode == "lowdin":
                T_orth = orth.lowdin(S_mol)
            else:
                from pyscf.lo.orth import orth_ao
                # mf is explicitly required here for occupations
                T_orth = orth_ao(mf, 'meta-lowdin', pre_orth_ao="ANO")

            T_inv = T_orth.T @ S_mol
            P_mol = T_inv @ P_mol @ T_inv.T
            S_mol = np.eye(mol.nao)
        elif mode == "gross":
            P_mol = (P_mol @ S_mol + S_mol @ P_mol) / 2
            S_mol = np.eye(mol.nao)

    print(f"| Total population (trace of PS): {np.trace(P_mol @ S_mol):.10f}")
    total_eff_pop = 0
    total_kept_pop = 0

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
            mol_at.verbose = 0
            mol_at.spin = atom_spins.get(sym, 0); mol_at.cart = mol.cart; mol_at.build()
            mf_at = scf.KS(mol_at) if "dft" in mf.__module__ else scf.HF(mol_at)
            if hasattr(mf, 'xc'): mf_at.xc = mf.xc
            mf_at.kernel()
            dm_at = mf_at.make_rdm1()
            P_at = dm_at[0] + dm_at[1] if dm_at.ndim == 3 else dm_at
            S_at = mf_at.get_ovlp()
            mat_block = S_at @ P_at @ S_at
            ovlp_block = S_at
            w, c, l_map, shell_map = spherical_average(mol_at, 0, mat_block, ovlp_block)
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
            w, c, l_map, shell_map = spherical_average(mol, ia, mat_block, ovlp_block)

        total_eff_pop += np.sum(w)

        final_idx = []
        unique_l = sorted(target_l_counts.keys())
        for l in unique_l:
            target_n_shells = target_l_counts[l]
            # Grab strictly the required number of top-occupied shells
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

def iao(mol, coeffs, source_basis='minao', pol_basis=None, heavy_only=False, full_basis=False):
    pmol = reference_mol(mol, polarized=(pol_basis is not None), pol_basis=pol_basis, source_basis=source_basis, heavy_only=heavy_only, full_basis=full_basis)
    return _do_iao(mol, coeffs, pmol=pmol), pmol

def fpiao(mol, coeffs, x=1.0, source_basis='minao', pol_basis='ano', heavy_only=False, full_basis=False):
    pmol = reference_mol(mol, polarized=True, pol_basis=pol_basis, source_basis=source_basis, x=x, heavy_only=heavy_only, full_basis=full_basis)
    return _do_iao(mol, coeffs, pmol=pmol), pmol

def autosad(mol, mf, free_atom=True, mode=None, polarized=False, heavy_only=False, full_basis=False):
    def do_autosad(mol, mf, C_occ, free_atom, mode, polarized, heavy_only, full_basis):
        w, effaos, pmol = get_effaos(mol, mf, free_atom=free_atom, mode=mode, polarized=polarized, heavy_only=heavy_only, full_basis=full_basis)
        return _do_iao(mol, C_occ, A_basis=effaos)
    if "U" in mf.__class__.__name__:
        ca, cb = mf.mo_coeff; return (do_autosad(mol, mf, ca[:, :mf.nelec[0]], free_atom, mode, polarized, heavy_only, full_basis),
                                      do_autosad(mol, mf, cb[:, :mf.nelec[1]], free_atom, mode, polarized, heavy_only, full_basis)), reference_mol(mol, heavy_only=heavy_only, full_basis=full_basis)
    return do_autosad(mol, mf, mf.mo_coeff[:, :mol.nelectron // 2], free_atom, mode, polarized, heavy_only, full_basis), reference_mol(mol, heavy_only=heavy_only, full_basis=full_basis)


def dfpiao(mol, coeffs, x=0.5, source_basis='minao', pol_basis='ano'):
    res_iao = iao(mol, coeffs, source_basis=source_basis)
    res_fpiao = fpiao(mol, coeffs, x=1.0, source_basis=source_basis, pol_basis=pol_basis)
    return res_iao, res_fpiao


def iao_nonorth(mol, coeffs, source_basis='minao', heavy_only=False, full_basis=False):
    pmol = reference_mol(mol, polarized=False, source_basis=source_basis, heavy_only=heavy_only, full_basis=full_basis)
    from pyscf.lo import iao as pyscf_iao
    C_iao_nonorth = pyscf_iao.iao(mol, coeffs, minao=pmol.basis)
    return C_iao_nonorth, pmol

