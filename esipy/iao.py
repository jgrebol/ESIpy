import sys
from pyscf import scf
import os
import numpy as np
import scipy.linalg
from pyscf import gto, scf
from pyscf.data import elements
from pyscf.lo import orth

def load_iao_dat_basis(file_path, symbol):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    basis = []; found_element = False; i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('ELEMENT'):
            parts = line.split()
            if len(parts) >= 3 and parts[2].upper() == symbol.upper():
                found_element = True; i += 1
                while i < len(lines) and 'NSHELL' not in lines[i]: i += 1
                if i < len(lines):
                    num_shells = int(lines[i].strip().split()[1]); i += 1
                    for _ in range(num_shells):
                        while i < len(lines) and 'SHELL' not in lines[i]: i += 1
                        if i >= len(lines): break
                        shell_parts = lines[i].strip().split()
                        l, nprim = int(shell_parts[1]), int(shell_parts[2]); i += 1
                        shell = [l]
                        for _ in range(nprim):
                            prim_parts = lines[i].strip().split()
                            shell.append([float(prim_parts[0]), float(prim_parts[1])]); i += 1
                        basis.append(shell)
                break
        i += 1
    if not found_element: raise ValueError(f'Element {symbol} not found')
    return basis

def _load_basis_wrapper(name, sym):
    if os.path.exists(name): return load_iao_dat_basis(name, sym)
    return gto.basis.load(name, sym)

def dump_matrix(path, a):
    with open(path, 'w') as f:
        f.write(f"{a.shape[0]} {a.shape[1]}\n")
        for row in a: f.write(" ".join([f"{x:24.16E}" for x in row]) + "\n")

def spherical_average(mol, ia, mat, overlap):
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
        p_frag = np.zeros((n_shells, n_shells)); s_frag = np.zeros((n_shells, n_shells))
        for i in range(n_shells):
            for j in range(n_shells):
                p_frag[i, j] = np.trace(mat[np.ix_(idx_reshaped[i], idx_reshaped[j])]) / degen
                s_frag[i, j] = np.trace(overlap[np.ix_(idx_reshaped[i], idx_reshaped[j])]) / degen
        try: e, v = scipy.linalg.eigh(p_frag, s_frag)
        except: e, v = scipy.linalg.eigh(p_frag)
        sort_idx = np.argsort(e)[::-1]; e, v = e[sort_idx], v[:, sort_idx]
        for iw in range(n_shells):
            val = e[iw]
            for m in range(degen):
                v_full = np.zeros(mat.shape[0])
                for i_shell in range(n_shells): v_full[idx_reshaped[i_shell, m]] = v[i_shell, iw]
                norm = np.sqrt(v_full.T @ overlap @ v_full)
                if norm > 1e-14: v_full /= norm
                all_w.append(val); l_map.append(l); shell_map.append(iw)
                all_c[:, current_col] = v_full; current_col += 1
    return np.array(all_w), all_c, np.array(l_map), np.array(shell_map)

def get_num_minbas_per_l(sym, polarized=False):
    # AUTOMATIC ADJUSTMENT using minao basis as reference, counting contracted orbitals
    basis = gto.basis.load('minao', sym)
    d = {}
    for shell in basis:
        l = shell[0]; n_cont = len(shell[1]) - 1
        d[l] = d.get(l, 0) + n_cont
    if polarized:
        z = elements.charge(sym)
        if z <= 2: d[1] = 1 # H, He: p
        elif z <= 10: # Li - Ne
            if 1 not in d: d[1] = 1 # Li, Be: p
            else: d[2] = 1 # B - Ne: d
        elif z <= 18: # Na - Ar
            if 2 not in d: d[2] = 1 # Na, Mg, Al...: d
            else: d[3] = 1
        else:
            if 3 not in d: d[3] = 1 # K...: f
            else: d[4] = 1
    return d

def _get_capped_target_l(mol, ia, polarized=False):
    sym = mol.atom_pure_symbol(ia)
    target_l = get_num_minbas_per_l(sym, polarized=polarized).copy()
    avail = {}
    for ib in range(mol.nbas):
        if mol.bas_atom(ib) == ia:
            l = mol.bas_angular(ib); avail[l] = avail.get(l, 0) + 1
    return {l: min(count, avail.get(l, 0)) for l, count in target_l.items()}

def get_num_minbas_ao(mol, ia, polarized=False, heavy_only=True):
    sym = mol.atom_pure_symbol(ia)
    is_pol = polarized and not (heavy_only and sym == 'H')
    target_l = _get_capped_target_l(mol, ia, polarized=is_pol)
    return sum(count * (2*l + 1) for l, count in target_l.items())

def get_reference_basis_dict(mol, source_basis='minao', pol_basis=None, x=1.0, heavy_only=False, full_basis=False):
    def get_minimal_part(sym, ia, source_basis_name):
        if full_basis: return _load_basis_wrapper(source_basis_name, sym), {}
        basis_source = _load_basis_wrapper(source_basis_name, sym)
        target_l = _get_capped_target_l(mol, ia, polarized=False)
        new_basis_atom, l_counts = [], {}
        for shell in basis_source:
            l = shell[0]; n_contractions = len(shell[1]) - 1
            current_l_count = l_counts.get(l, 0); needed_l_count = target_l.get(l, 0)
            if current_l_count < needed_l_count:
                n_to_take = min(n_contractions, needed_l_count - current_l_count)
                if n_to_take == n_contractions: new_basis_atom.append(shell)
                else:
                    new_shell = [l]
                    for prim in shell[1:]: new_shell.append(prim[:n_to_take+1])
                    new_basis_atom.append(new_shell)
                l_counts[l] = current_l_count + n_to_take
        return new_basis_atom, target_l

    def get_polarization_part(sym, ia, pol_basis_name, target_l, x=1.0):
        if pol_basis_name == 'working': basis_pol = mol._basis[sym]
        elif isinstance(pol_basis_name, str): basis_pol = _load_basis_wrapper(pol_basis_name, sym)
        else: return []
        target_pol_l = _get_capped_target_l(mol, ia, polarized=True)
        new_pol_atom, l_counts = [], {}
        for shell in basis_pol:
            l = shell[0]
            if l in target_pol_l and l not in target_l:
                n_contractions = len(shell[1]) - 1
                current_l_count = l_counts.get(l, 0); needed_l_count = target_pol_l[l]
                if current_l_count < needed_l_count:
                    n_to_take = min(n_contractions, needed_l_count - current_l_count)
                    new_shell = [l]
                    for prim in shell[1:]:
                        scaled_exp = prim[0] * x; new_shell.append([scaled_exp] + list(prim[1:n_to_take+1]))
                    new_pol_atom.append(new_shell); l_counts[l] = current_l_count + n_to_take
        return new_pol_atom

    ref_basis = {}
    for ia in range(mol.natm):
        sym = mol.atom_pure_symbol(ia)
        if sym not in ref_basis:
            min_basis, target_l = get_minimal_part(sym, ia, source_basis)
            pol_basis_list = get_polarization_part(sym, ia, pol_basis, target_l, x=x) if not full_basis else []
            ref_basis[sym] = min_basis + pol_basis_list
    return ref_basis

def reference_mol(mol, polarized=False, pol_basis=None, source_basis='minao', x=1.0, heavy_only=False, full_basis=False):
    if not isinstance(mol, gto.Mole): mol = getattr(mol, 'pyscf_mol', getattr(mol, 'mol', mol))
    if polarized and pol_basis is None: pol_basis = 'ano'
    elif not polarized: pol_basis = None
    ref_basis = get_reference_basis_dict(mol, source_basis=source_basis, pol_basis=pol_basis, x=x, heavy_only=heavy_only, full_basis=full_basis)
    if heavy_only:
        for sym in ref_basis:
            if sym == 'H': ref_basis[sym] = [sh for sh in ref_basis[sym] if sh[0] == 0]
    pmol = gto.Mole()
    pmol.atom = mol.atom; pmol.cart = mol.cart; pmol.unit = mol.unit; pmol.basis = ref_basis
    pmol.charge = mol.charge; pmol.spin = mol.spin; pmol.build()
    return pmol

def get_effaos(mol, coeffs, free_atom=True, mode='net', polarized=False, heavy_only=False, full_basis=False, x=1.0, mf=None):
    if not isinstance(mol, gto.Mole): mol = getattr(mol, 'pyscf_mol', getattr(mol, 'mol', mol))
    pmol = reference_mol(mol, polarized=polarized, heavy_only=heavy_only, full_basis=full_basis, x=x)
    minbas_total = pmol.nao; pmol_aoslices = pmol.aoslice_by_atom()
    veps_block = np.zeros((mol.nao, minbas_total)); vaps_diag = []
    S_mol = mol.intor("int1e_ovlp"); P_mol = coeffs @ coeffs.T; T_orth = None
    if not free_atom:
        if mode in ["lowdin", "meta-lowdin", "ml", "mlowdin", "meta_lowdin", "metalowdin", "nao"]:
            from pyscf import lo
            method = "nao" if mode == "nao" else ("lowdin" if mode in ["lowdin", "ml", "mlowdin"] else "meta-lowdin")
            T_orth = lo.orth_ao(mf if mode == "nao" else mol, method=method, s=S_mol)
            T_inv = np.linalg.inv(T_orth); P_mol = T_inv @ P_mol @ T_inv.T; S_mol = np.eye(mol.nao)
        elif mode == "gross":
            PS = P_mol @ S_mol; P_mol = (PS + PS.T) * 0.5; S_mol = np.eye(mol.nao)
    aoslices = mol.aoslice_by_atom(); col_idx = 0
    for ia in range(mol.natm):
        sym = mol.atom_pure_symbol(ia); p0, p1 = aoslices[ia, 2], aoslices[ia, 3]
        p0_ref, p1_ref = pmol_aoslices[ia, 2], pmol_aoslices[ia, 3]
        n_target = p1_ref - p0_ref
        is_pol = polarized and not (heavy_only and sym == "H")
        target_l_counts = _get_capped_target_l(mol, ia, polarized=is_pol)
        if free_atom:
            atom_spins = {'H': 1, 'He': 0, 'Li': 1, 'Be': 0, 'B': 1, 'C': 2, 'N': 3, 'O': 2, 'F': 1, 'Ne': 0, 'Na': 1, 'Mg': 0, 'Al': 1, 'Si': 2, 'P': 3, 'S': 2, 'Cl': 1, 'Ar': 0, 'K': 1, 'Ca': 0, 'Sc': 1, 'Ti': 2, 'V': 3, 'Cr': 6, 'Mn': 5, 'Fe': 4, 'Co': 3, 'Ni': 2, 'Cu': 1, 'Zn': 0, 'Ga': 1, 'Ge': 2, 'As': 3, 'Se': 2, 'Br': 1, 'Kr': 0, 'Rb': 1, 'Sr': 0, 'Y': 1, 'Zr': 2, 'Nb': 5, 'Mo': 6, 'Tc': 5, 'Ru': 4, 'Rh': 3, 'Pd': 0, 'Ag': 1, 'Cd': 0, 'In': 1, 'Sn': 2, 'Sb': 3, 'Te': 2, 'I': 1, 'Xe': 0, 'Cs': 1, 'Ba': 0, 'La': 1, 'Ce': 2, 'Pr': 3, 'Nd': 4, 'Pm': 5, 'Sm': 6, 'Eu': 7, 'Gd': 8, 'Tb': 5, 'Dy': 4, 'Ho': 3, 'Er': 2, 'Tm': 1, 'Yb': 0, 'Lu': 1, 'Hf': 2, 'Ta': 3, 'W': 4, 'Re': 5, 'Os': 4, 'Ir': 3, 'Pt': 2, 'Au': 1, 'Hg': 0, 'Tl': 1, 'Pb': 2, 'Bi': 3, 'Po': 2, 'At': 1, 'Rn': 0, 'Fr': 1, 'Ra': 0, 'Ac': 1, 'Th': 2, 'Pa': 3, 'U': 4, 'Np': 5, 'Pu': 6, 'Am': 7, 'Cm': 8, 'Bk': 5, 'Cf': 4, 'Es': 3, 'Fm': 2, 'Md': 1, 'No': 0, 'Lr': 1, 'Rf': 2, 'Db': 3, 'Sg': 4, 'Bh': 5, 'Hs': 4, 'Mt': 3, 'Ds': 2, 'Rg': 1, 'Cn': 0, 'Nh': 1, 'Fl': 2, 'Mc': 3, 'Lv': 2, 'Ts': 1, 'Og': 0}
            atom_mol = gto.M(atom=f"{sym} 0 0 0", basis=mol.basis, spin=atom_spins.get(sym, elements.charge(sym) % 2), charge=0, cart=mol.cart)
            atom_mf = scf.RHF(atom_mol) if atom_mol.spin == 0 else scf.UHF(atom_mol)
            atom_mf.verbose = 0; atom_mf.kernel()
            S_at = atom_mol.intor('int1e_ovlp'); P_at = atom_mf.make_rdm1()
            if P_at.ndim == 3: P_at = P_at[0] + P_at[1]
            mat_block = S_at @ P_at @ S_at; ovlp_block = S_at
            w, c, l_map, shell_map = spherical_average(atom_mol, 0, mat_block, ovlp_block)
        else:
            if mode == "net": mat_block = S_mol[p0:p1, p0:p1] @ P_mol[p0:p1, p0:p1] @ S_mol[p0:p1, p0:p1]; ovlp_block = S_mol[p0:p1, p0:p1]
            elif mode == "gross" or mode in ["lowdin", "meta-lowdin", "ml", "mlowdin", "meta_lowdin", "metalowdin", "nao"]: mat_block = P_mol[p0:p1, p0:p1]; ovlp_block = np.eye(p1 - p0)
            elif mode in ["symmetric", "sym"]: mat_block = (P_mol[p0:p1, p0:p1] @ S_mol[p0:p1, p0:p1] + S_mol[p0:p1, p0:p1] @ P_mol[p0:p1, p0:p1]) / 2; ovlp_block = np.eye(p1 - p0)
            elif mode == "sps": mat_block = (S_mol @ P_mol @ S_mol)[p0:p1, p0:p1]; ovlp_block = np.eye(p1 - p0)
            elif mode == "spsa": mat_block = (S_mol @ P_mol @ S_mol)[p0:p1, p0:p1]; ovlp_block = S_mol[p0:p1, p0:p1]
            else: mat_block = P_mol[p0:p1, p0:p1]; ovlp_block = S_mol[p0:p1, p0:p1]
            w, c, l_map, shell_map = spherical_average(mol, ia, mat_block, ovlp_block)
        final_idx = []
        unique_l = sorted(target_l_counts.keys())
        for l in unique_l:
            target_n_shells = target_l_counts[l]
            idx_keep = np.where((l_map == l) & (shell_map < target_n_shells))[0]
            final_idx.extend(idx_keep)
        final_idx = np.sort(final_idx); w_keep = w[final_idx]; c_keep = c[:, final_idx]
        veps_block[p0:p1, col_idx: col_idx + n_target] = c_keep
        vaps_diag.extend(w_keep); col_idx += n_target
    if T_orth is not None: veps_block = T_orth @ veps_block
    return np.array(vaps_diag), veps_block, pmol

def _do_iao(mol, coeffs, pmol=None, A_basis=None, heavy_only=False):
    if not isinstance(mol, gto.Mole): mol = getattr(mol, 'pyscf_mol', getattr(mol, 'mol', mol))
    S1 = mol.intor('int1e_ovlp')
    if A_basis is not None:
        A_tilde = A_basis; S12 = S1 @ A_tilde; S2 = A_tilde.T @ S12
    else:
        from pyscf.gto.mole import intor_cross
        S12 = intor_cross('int1e_ovlp', mol, pmol); A_tilde = scipy.linalg.solve(S1, S12, assume_a='pos'); S2 = pmol.intor_symmetric('int1e_ovlp')
    C_min = scipy.linalg.solve(S2, S12.T @ coeffs, assume_a='pos')
    C_proj = orth.vec_lowdin(A_tilde @ C_min, S1)
    P_occ_A = coeffs @ (coeffs.T @ S12)
    P_proj_A = C_proj @ (C_proj.T @ S12)
    P_occ_P_proj_A = coeffs @ (coeffs.T @ (S1 @ P_proj_A))
    IAO_nonorth = A_tilde + 2 * P_occ_P_proj_A - P_occ_A - P_proj_A
    return orth.vec_lowdin(IAO_nonorth, S1)

def iao(mol, coeffs, source_basis='minao', heavy_only=False, full_basis=False):
    pmol = reference_mol(mol, polarized=False, source_basis=source_basis, heavy_only=heavy_only, full_basis=full_basis)
    return _do_iao(mol, coeffs, pmol=pmol), pmol

def fpiao(mol, coeffs, x=1.0, source_basis='minao', pol_basis='ano', heavy_only=True, full_basis=False):
    pmol = reference_mol(mol, polarized=True, pol_basis=pol_basis, source_basis=source_basis, x=x, heavy_only=heavy_only, full_basis=full_basis)
    return _do_iao(mol, coeffs, pmol=pmol), pmol

def autosad(mol, coeffs, polarized=False, heavy_only=False, full_basis=False, x=1.0, mf=None):
    w, A, pmol = get_effaos(mol, coeffs, free_atom=True, polarized=polarized, heavy_only=heavy_only, full_basis=full_basis, x=x, mf=mf)
    return _do_iao(mol, coeffs, A_basis=A, heavy_only=heavy_only), pmol

def effao(mol, coeffs, mode='net', polarized=False, heavy_only=False, full_basis=False, x=1.0, mf=None):
    w, A, pmol = get_effaos(mol, coeffs, free_atom=False, mode=mode, polarized=polarized, heavy_only=heavy_only, full_basis=full_basis, x=x, mf=mf)
    return _do_iao(mol, coeffs, A_basis=A, heavy_only=heavy_only), pmol

def fpiao_effao(mol, coeffs, x=1.0, mode='nao', pol_basis='ano', heavy_only=True, full_basis=False, mf=None):
    w, A_min, pmol_min = get_effaos(mol, coeffs, free_atom=False, mode=mode, polarized=False, heavy_only=heavy_only, full_basis=full_basis, mf=mf)
    pmol_pol = reference_mol(mol, polarized=True, pol_basis=pol_basis, source_basis='minao', x=x, heavy_only=heavy_only, full_basis=full_basis)
    S1 = mol.intor('int1e_ovlp')
    from pyscf.gto.mole import intor_cross
    S12 = intor_cross('int1e_ovlp', mol, pmol_pol)
    A_ano_all = scipy.linalg.solve(S1, S12, assume_a='pos')
    A_total = np.zeros((mol.nao, pmol_pol.nao))
    aos_min = pmol_min.aoslice_by_atom(); aos_pol = pmol_pol.aoslice_by_atom()
    for ia in range(mol.natm):
        p0_m, p1_m = aos_min[ia, 2], aos_min[ia, 3]; p0_p, p1_p = aos_pol[ia, 2], aos_pol[ia, 3]
        n_min = p1_m - p0_m
        A_total[:, p0_p : p0_p + n_min] = A_min[:, p0_m:p1_m]
        A_total[:, p0_p + n_min : p1_p] = A_ano_all[:, p0_p + n_min : p1_p]
    return _do_iao(mol, coeffs, A_basis=A_total, heavy_only=heavy_only), pmol_pol

def peiao(mol, coeffs, mode='nao', heavy_only=True, full_basis=False, mf=None):
    # Polarized-Effao-IAO: Both minimal and polarization parts from effaos of the actual basis
    w, A_both, pmol = get_effaos(mol, coeffs, free_atom=False, mode=mode, polarized=True, heavy_only=heavy_only, full_basis=full_basis, mf=mf)
    return _do_iao(mol, coeffs, A_basis=A_both, heavy_only=heavy_only), pmol
