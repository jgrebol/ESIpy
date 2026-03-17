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

def get_num_minbas_per_l(sym, polarized=False):
    """Returns the number of physical shells required per L."""
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
    """Returns the total number of AOs in the minimal basis for atom ia."""
    sym = mol.atom_pure_symbol(ia)
    target_l = get_num_minbas_per_l(sym, polarized=polarized)
    total = 0
    ao_loc = mol.ao_loc_nr()
    atom_shells = [ib for ib in range(mol.nbas) if mol.bas_atom(ib) == ia]
    for l, count in target_l.items():
        degen = 2 * l + 1
        for ib in atom_shells:
            if mol.bas_angular(ib) == l:
                degen = ao_loc[ib+1] - ao_loc[ib]; break
        total += count * degen
    return total

def get_reference_basis_dict(mol, source_basis='minao', pol_basis=None, x=1.0):
    """Helper function to get the reference basis set for IAO."""
    
    def get_minimal_part(sym, source_basis_name):
        # Truncate to minimal basis size
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
        else:
            return []

        # Target polarization L shells
        target_pol_l = get_num_minbas_per_l(sym, polarized=True)
        # Only keep L that are in target_pol_l but NOT in target_l
        new_pol_atom, l_counts = [], {}
        for shell in basis_pol:
            l = shell[0]
            if l in target_pol_l and l not in target_l:
                count = l_counts.get(l, 0)
                if count < target_pol_l[l]:
                    if x != 1.0:
                        scaled_shell = [l]
                        for prim in shell[1:]:
                            new_prim = [prim[0] * x] + list(prim[1:])
                            scaled_shell.append(new_prim)
                        new_pol_atom.append(scaled_shell)
                    else:
                        new_pol_atom.append(shell)
                    l_counts[l] = count + 1
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
    """Builds a reference Mole object for IAO."""
    from pyscf.lo import iao as pyscf_iao
    if hasattr(mol, 'pyscf_mol'):
        mol = mol.pyscf_mol
    
    # If polarized is True but pol_basis is None, use 'minao' polarization
    if polarized and pol_basis is None:
        pol_basis = 'minao'
    elif not polarized:
        pol_basis = None

    ref_basis = get_reference_basis_dict(mol, source_basis=source_basis, pol_basis=pol_basis, x=x)
    return pyscf_iao.reference_mol(mol, minao=ref_basis)

def get_effaos(mol, mf, free_atom=True, mode=None, polarized=False):
    """
    Builds effective Atomic Orbitals (eff-AOs) for all IAO variants.
    """
    pmol = reference_mol(mol, polarized=polarized)
    minbas_total = pmol.nao
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
            if mode == "lowdin": T = orth.lowdin(S_mol) # T is S^-1/2
            else:
                from pyscf.lo.orth import orth_ao
                T = orth_ao(mf, 'meta_lowdin', pre_orth_ao="MINAO")
            # User wants (T^-1 P T^-1)^A. If T = S^-1/2, then T^-1 = S^1/2.
            # P_lowdin = S^1/2 P S^1/2.
            # We can get this by T_inv = S^1/2.
            # Or simpler: P_mol = inv(T) @ P_mol @ inv(T)
            T_inv = scipy.linalg.sqrtm(S_mol)
            P_mol = T_inv @ P_mol @ T_inv
            S_mol = np.eye(mol.nao)
        elif mode == "gross":
            # P_gross = (PS + SP) / 2
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
            # For free atom, eigenvalues of free-atom density matrix.
            # Usually means S P S c = S c lambda
            mat_block = S_at @ P_at @ S_at
            ovlp_block = S_at
            w, c, l_map = spherical_average(mol_at, 0, mat_block, ovlp_block)
        else:
            # 1. Define Property Matrix (mat_block) and Metric (ovlp_block)
            if mode == "net":
                # S^A P^A S^A c^A = S^A c^A lambda^A
                Sa, Pa = S_mol[p0:p1, p0:p1], P_mol[p0:p1, p0:p1]
                mat_block = Sa @ Pa @ Sa
                ovlp_block = Sa
            elif mode == "gross":
                # ( (PS + SP)/2 )^A c^A = c^A lambda^A
                # S_mol is already I here due to global transform
                mat_block = P_mol[p0:p1, p0:p1]
                ovlp_block = np.eye(p1 - p0)
            elif mode in ["lowdin", "meta-lowdin"]:
                # ( T^-1 P T^-1 )^A c^A = c^A lambda^A
                # S_mol is already I here due to global transform
                mat_block = P_mol[p0:p1, p0:p1]
                ovlp_block = np.eye(p1 - p0)
            elif mode == "sym":
                # ( (P^A S^A + S^A P^A)/2 ) c^A = c^A lambda^A
                Pa, Sa = P_mol[p0:p1, p0:p1], S_mol[p0:p1, p0:p1]
                mat_block = (Pa @ Sa + Sa @ Pa) / 2
                ovlp_block = np.eye(p1 - p0)
            elif mode == "sps":
                # (S P S)^A c^A = c^A lambda^A
                mat_block = (S_mol @ P_mol @ S_mol)[p0:p1, p0:p1]
                ovlp_block = np.eye(p1 - p0)
            elif mode == "spsa":
                # (S P S)^A c^A = S^A c^A lambda^A
                mat_block = (S_mol @ P_mol @ S_mol)[p0:p1, p0:p1]
                ovlp_block = S_mol[p0:p1, p0:p1]
            else:
                # Default fallback
                mat_block = P_mol[p0:p1, p0:p1]
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
            # Fallback for polarized shells not in basis? 
            if degen == 0: degen = 2 * l + 1
            selected.extend(l_indices[:count * degen])
        
        selected = sorted(selected)
        veps_block[p0:p1, col_idx : col_idx + n_target] = c[:, selected]
        vaps_diag.extend(w[selected])
        col_idx += n_target

    return np.array(vaps_diag), veps_block, pmol

def _do_iao(mol, coeffs, pmol=None, A_basis=None):
    """
    Core IAO construction. 
    Can take either a pmol (reference Mole) or A_basis (basis in working basis).
    """
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
        # Use PySCF's construction for standard cases
        C_iao_nonorth = pyscf_iao.iao(mol, coeffs, minao=pmol.basis)
        return orth.vec_lowdin(C_iao_nonorth, S1)

def iao(mol, coeffs, source_basis='minao', pol_basis=None):
    """Standard IAO construction."""
    pmol = reference_mol(mol, polarized=(pol_basis is not None), pol_basis=pol_basis, source_basis=source_basis)
    return _do_iao(mol, coeffs, pmol=pmol), pmol

def piao(mol, coeffs, source_basis='minao', pol_basis='working'):
    """Polarized IAO."""
    pmol = reference_mol(mol, polarized=True, pol_basis=pol_basis, source_basis=source_basis)
    return _do_iao(mol, coeffs, pmol=pmol), pmol

def fpiao(mol, coeffs, x=1.0, source_basis='minao', pol_basis='ano'):
    """Polarized IAO using polarization from specified basis (default ANO)."""
    pmol = reference_mol(mol, polarized=True, pol_basis=pol_basis, source_basis=source_basis, x=x)
    return _do_iao(mol, coeffs, pmol=pmol), pmol

def autosad(mol, mf, free_atom=True, mode=None, polarized=False):
    """IAO-AutoSAD and IAO-effAO wrapper."""
    def do_autosad(mol, mf, C_occ, free_atom, mode, polarized):
        w, effaos, pmol = get_effaos(mol, mf, free_atom=free_atom, mode=mode, polarized=polarized)
        return _do_iao(mol, C_occ, A_basis=effaos)
    
    if "U" in mf.__class__.__name__:
        ca, cb = mf.mo_coeff; return (do_autosad(mol, mf, ca[:, :mf.nelec[0]], free_atom, mode, polarized), 
                                      do_autosad(mol, mf, cb[:, :mf.nelec[1]], free_atom, mode, polarized)), reference_mol(mol)
    return do_autosad(mol, mf, mf.mo_coeff[:, :mol.nelectron // 2], free_atom, mode, polarized), reference_mol(mol)
