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
        # Shells of this L for this atom
        shell_indices = [ib for ib in atom_shell_indices if mol.bas_angular(ib) == l]
        first_ib = shell_indices[0]
        comp_degen = ao_loc[first_ib+1] - ao_loc[first_ib] # 5 for D-sph, 6 for D-cart
        n_shells = len(shell_indices)
        
        # Build reduced radial matrices (n_shells x n_shells)
        mat_red = np.zeros((n_shells, n_shells))
        ovlp_red = np.zeros((n_shells, n_shells))
        
        for i, ib in enumerate(shell_indices):
            for j, jb in enumerate(shell_indices):
                p0, p1 = ao_loc[ib] - atom_start, ao_loc[ib+1] - atom_start
                q0, q1 = ao_loc[jb] - atom_start, ao_loc[jb+1] - atom_start
                
                # Average the block (Trace / dimensionality)
                mat_red[i, j] = np.trace(mat[p0:p1, q0:q1]) / comp_degen
                ovlp_red[i, j] = np.trace(overlap[p0:p1, q0:q1]) / comp_degen
        
        # Solve GEVP for the radial part: M_red * c = O_red * c * lambda
        try:
            # S^-1/2 transformation for stability
            e, v = scipy.linalg.eigh(ovlp_red)
            e[e < 1e-14] = 1e-14
            s_inv_half = v @ np.diag(1.0 / np.sqrt(e)) @ v.T
            w, c_prime = scipy.linalg.eigh(s_inv_half @ mat_red @ s_inv_half)
            c_red = s_inv_half @ c_prime
        except Exception:
            w, c_red = scipy.linalg.eigh(mat_red)

        # Sort descending
        idx = np.argsort(w)[::-1]
        w, c_red = w[idx], c_red[:, idx]
        
        # Expand back to the full AO basis of the atom
        for iw, val in enumerate(w):
            for m in range(comp_degen):
                v_full = np.zeros(mat.shape[0])
                for i, ib in enumerate(shell_indices):
                    p_rel = ao_loc[ib] - atom_start
                    v_full[p_rel + m] = c_red[i, iw]
                
                # Ensure S-orthonormality in the atomic basis
                norm = np.sqrt(v_full.T @ overlap @ v_full)
                if norm > 1e-14: v_full /= norm
                
                all_w.append(val)
                l_map.append(l)
                all_c[:, current_col] = v_full
                current_col += 1

    return np.array(all_w), all_c, np.array(l_map)

def get_num_minbas_per_l(sym):
    """Returns the number of physical shells required per L."""
    z = elements.charge(sym)
    if z <= 2:   return {0: 1}  # 1s
    if z <= 10:  return {0: 2, 1: 1}  # 1s, 2s, 2p
    if z <= 18:  return {0: 3, 1: 2}  # 3s, 3p
    if z <= 36:  return {0: 4, 1: 3, 2: 1}  # 4s, 4p, 3d
    if z <= 54:  return {0: 5, 1: 4, 2: 2}  # 5s, 5p, 4d
    if z <= 86:  return {0: 6, 1: 5, 2: 3, 3: 1}
    if z <= 118: return {0: 7, 1: 6, 2: 4, 3: 2}
    raise NotImplementedError(f"Minimal basis not defined for: {sym}")

def get_num_minbas_ao(mol, ia):
    """Returns the total number of AOs in the minimal basis for atom ia."""
    sym = mol.atom_pure_symbol(ia)
    target_l = get_num_minbas_per_l(sym)
    total = 0
    ao_loc = mol.ao_loc_nr()
    # Find first shell of this atom to check degen (Cartesian vs Spherical)
    atom_shells = [ib for ib in range(mol.nbas) if mol.bas_atom(ib) == ia]
    
    for l, count in target_l.items():
        # Find actual degen for this L in this basis
        degen = 2 * l + 1 # Default
        for ib in atom_shells:
            if mol.bas_angular(ib) == l:
                degen = ao_loc[ib+1] - ao_loc[ib]
                break
        total += count * degen
    return total

def get_effaos(mol, mf, free_atom=True, mode=None):
    """
    Builds effective Atomic Orbitals (eff-AOs) for all IAO variants.
    """
    minbas_total = sum(get_num_minbas_ao(mol, i) for i in range(mol.natm))
    veps_block = np.zeros((mol.nao, minbas_total))
    vaps_diag = []
    
    S_mol = mol.intor("int1e_ovlp")
    if "U" in mf.__class__.__name__:
        dm = mf.make_rdm1(ao_repr=True); P_mol = dm[0] + dm[1] if dm.ndim == 3 else dm
    else:
        P_mol = mf.make_rdm1(ao_repr=True)

    # Global transformations for specific modes
    if not free_atom:
        if mode in ["lowdin", "meta-lowdin"]:
            if mode == "lowdin": T = orth.lowdin(S_mol)
            else:
                from pyscf.lo.orth import orth_ao
                T = orth_ao(mf, 'meta_lowdin', pre_orth_ao="ANO")
            T_inv = scipy.linalg.inv(T)
            # P' = T^-1 P T^-1
            P_mol = T_inv @ P_mol @ T_inv.T
        elif mode == "gross":
            # P' = (PS + SP) / 2
            P_mol = (P_mol @ S_mol + S_mol @ P_mol) / 2

    aoslices = mol.aoslice_by_atom()
    col_idx = 0
    atom_spins = {'H':1,'He':0,'Li':1,'Be':0,'B':1,'C':2,'N':3,'O':2,'F':1,'Ne':0,'Na':1,'Mg':0,'Al':1,'Si':2,'P':3,'S':2,'Cl':1,'Ar':0}

    for ia in range(mol.natm):
        sym = mol.atom_pure_symbol(ia)
        p0, p1 = aoslices[ia, 2], aoslices[ia, 3]
        target_l_counts = get_num_minbas_per_l(sym)
        n_target = get_num_minbas_ao(mol, ia)

        if free_atom:
            mol_at = gto.Mole(); mol_at.atom = f"{sym} 0 0 0"; mol_at.basis = {sym: mol._basis[sym]}
            mol_at.spin = atom_spins.get(sym, 0); mol_at.cart = mol.cart; mol_at.build()
            mf_at = scf.KS(mol_at) if "dft" in mf.__module__ else scf.HF(mol_at)
            if hasattr(mf, 'xc'): mf_at.xc = mf.xc
            mf_at.kernel()
            dm_at = mf_at.make_rdm1()
            P_at = dm_at[0] + dm_at[1] if dm_at.ndim == 3 else dm_at
            S_at = mf_at.get_ovlp()
            # IAO-AUTOSAD uses SPS of free atom for selection
            w, c, l_map = spherical_average(mol_at, 0, S_at @ P_at @ S_at, S_at)
        else:
            # 1. Define Property Matrix (mat_block)
            if mode == "net":
                # S^A P^A S^A
                Sa, Pa = S_mol[p0:p1, p0:p1], P_mol[p0:p1, p0:p1]
                mat_block = Sa @ Pa @ Sa
            elif mode in ["sps", "spsa"]:
                # (S P S)^A
                mat_block = (S_mol @ P_mol @ S_mol)[p0:p1, p0:p1]
            elif mode == "sym":
                # (P^A S^A + S^A P^A) / 2
                Pa, Sa = P_mol[p0:p1, p0:p1], S_mol[p0:p1, p0:p1]
                mat_block = (Pa @ Sa + Sa @ Pa) / 2
            else:
                # GROSS, LOWDIN, ML, default
                mat_block = P_mol[p0:p1, p0:p1]

            # 2. Define Metric (ovlp_block)
            if mode in ["lowdin", "meta-lowdin", "gross", "sym", "sps"]:
                ovlp_block = np.eye(p1 - p0)
            else:
                # net, spsa, and default use the local overlap block
                ovlp_block = S_mol[p0:p1, p0:p1]
            
            w, c, l_map = spherical_average(mol, ia, mat_block, ovlp_block)

        # Selection based on L shell requirements
        selected = []
        ao_loc = mol.ao_loc_nr()
        for l, count in target_l_counts.items():
            l_indices = np.where(l_map == l)[0]
            # Actual degen for this L
            degen = 0
            for ib in range(mol.nbas):
                if mol.bas_atom(ib) == ia and mol.bas_angular(ib) == l:
                    degen = ao_loc[ib+1] - ao_loc[ib]; break
            selected.extend(l_indices[:count * degen])
        
        selected = sorted(selected)
        veps_block[p0:p1, col_idx : col_idx + n_target] = c[:, selected]
        vaps_diag.extend(w[selected])
        col_idx += n_target

    return np.array(vaps_diag), veps_block

def iao(mol, pmol, coeffs):
    """Standard IAO construction using Knizia formula."""
    print("hola")
    from pyscf.gto.mole import intor_cross
    S1, S2 = mol.intor('int1e_ovlp'), pmol.intor('int1e_ovlp')
    S12 = intor_cross('int1e_ovlp', mol, pmol)
    C_occ = coeffs[:, :mol.nelectron // 2]
    A_tilde = scipy.linalg.solve(S1, S12, assume_a='pos')
    C_min = scipy.linalg.solve(S2, S12.T @ C_occ, assume_a='pos')
    C_proj = orth.vec_lowdin(A_tilde @ C_min, S1)
    P_occ_A = C_occ @ (C_occ.T @ S12); P_proj_A = C_proj @ (C_proj.T @ S12)
    P_occ_P_proj_A = C_occ @ (C_occ.T @ (S1 @ P_proj_A))
    IAO_nonorth = A_tilde + 2 * P_occ_P_proj_A - P_occ_A - P_proj_A
    return orth.vec_lowdin(IAO_nonorth, S1)

def autosad(mol, mf, free_atom=True, mode=None):
    """IAO-AutoSAD and IAO-effAO wrapper."""
    def do_autosad(mol, mf, C_occ, free_atom, mode):
        S1 = mol.intor('int1e_ovlp')
        w, effaos = get_effaos(mol, mf, free_atom=free_atom, mode=mode)
        S12_eff = S1 @ effaos; S2_eff = effaos.T @ S1 @ effaos
        C_min = scipy.linalg.solve(S2_eff, S12_eff.T @ C_occ, assume_a='pos')
        C_proj = orth.vec_lowdin(effaos @ C_min, S1)
        P_occ_A = C_occ @ (C_occ.T @ S12_eff); P_proj_A = C_proj @ (C_proj.T @ S12_eff)
        P_occ_P_proj_A = C_occ @ (C_occ.T @ (S1 @ P_proj_A))
        IAO_nonorth = effaos + 2 * P_occ_P_proj_A - P_occ_A - P_proj_A
        return orth.vec_lowdin(IAO_nonorth, S1)
    if "U" in mf.__class__.__name__:
        ca, cb = mf.mo_coeff; return do_autosad(mol, mf, ca[:, :mf.nelec[0]], free_atom, mode), do_autosad(mol, mf, cb[:, :mf.nelec[1]], free_atom, mode)
    return do_autosad(mol, mf, mf.mo_coeff[:, :mol.nelectron // 2], free_atom, mode)
