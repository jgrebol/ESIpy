import numpy as np
import re
from pyscf.lo import nao
from pyscf.lo.orth import lowdin
from pyscf import scf

from esipy.tools import save_file, format_partition, get_natorbs, build_eta

def make_aoms(mol, mf, partition, myhf=None, save=None):
    """
    Build the Atomic Overlap Matrices (AOMs) in the Molecular Orbitals basis. 
    """

    partition_label = format_partition(partition)
    
    try:
        S = mf.get_ovlp()
    except:
        S = mol.intor('int1e_ovlp')
    
    def get_iao_aoms(p_type, c, current_mf, w_override=None, current_myhf=None, c_full=None):
        try:
            import iao_dump as iao_mod
        except ImportError:
            import esipy.iao as iao_mod
        
        iao, fpiao = iao_mod.iao, iao_mod.fpiao
        effao, autosad, wiao = iao_mod.effao, iao_mod.autosad, iao_mod.wiao
        reference_mol = iao_mod.reference_mol

        p_type = p_type.lower()
        if p_type.startswith("iao_"):
            p_type = "iao " + p_type[4:]
            
        m_w = re.search(r"\(([\d\.]+)\)", p_type)
        if m_w:
            string_w = float(m_w.group(1))
            p_type_clean = re.sub(r"\([\d\.]+\)", "", p_type).replace("  ", " ").strip()
        else:
            string_w = None
            p_type_clean = p_type.strip()

        # Extract heavy_only from obj if available or default to True
        local_heavy_only = True
        if "$hpol" in p_type_clean:
            local_heavy_only = False
            p_type_clean = p_type_clean.replace("$hpol", "").strip()

        p_parts = p_type_clean.split()
        p_base = p_parts[0]
        if p_base == "iao-basis":
            p_base = "iao"
            
        # Defaults
        iaoref = 'minao'
        ref_bas = p_parts[1] if len(p_parts) > 1 else iaoref
        weight = 0.5
        local_w = w_override if w_override is not None else (string_w if string_w is not None else (weight if weight is not None else 0.5))

        # IAO construction for Natural Orbitals: 
        # use first N_ref NOs for construction, but project all NOs
        is_no = False
        if hasattr(current_mf, 'mo_occ'):
            occ_tmp = np.asarray(current_mf.mo_occ)
            if np.any((occ_tmp > 1e-6) & (np.abs(occ_tmp - 1.0) > 1e-6) & (np.abs(occ_tmp - 2.0) > 1e-6)):
                is_no = True

        def get_c_src(is_pol):
            src = c_full if c_full is not None else c
            if not is_no: return src
            pmol_ref = reference_mol(mol, polarized=is_pol, source_basis=ref_bas, heavy_only=local_heavy_only, full_basis=False)
            return src[:, :pmol_ref.nao]

        if p_base == "dfpiao":
            mode_map = {'minao': 'nao', 'low': 'lowdin', 'ml': 'meta-lowdin', 'gross': 'gross'}
            actual_mode = mode_map.get(ref_bas.lower(), ref_bas)
            aom_iao = get_iao_aoms(f"iao-effao-{actual_mode}", c, current_mf, current_myhf=current_myhf, c_full=c_full)
            aom_fpiao = get_iao_aoms(f"fpiao {ref_bas}", c, current_mf, w_override=1.0, current_myhf=current_myhf, c_full=c_full)
            return [local_w * aom_iao[i] + (1.0 - local_w) * aom_fpiao[i] for i in range(mol.natm)]
        elif p_base == "dpeiao":
            mode_map = {'minao': 'nao', 'low': 'lowdin', 'ml': 'meta-lowdin', 'gross': 'gross'}
            actual_mode = mode_map.get(ref_bas.lower(), ref_bas)
            aom_iao = get_iao_aoms(f"iao-effao-{actual_mode}", c, current_mf, current_myhf=current_myhf, c_full=c_full)
            aom_peiao = get_iao_aoms(f"peiao {ref_bas}", c, current_mf, current_myhf=current_myhf, c_full=c_full)
            return [local_w * aom_iao[i] + (1.0 - local_w) * aom_peiao[i] for i in range(mol.natm)]
        elif p_base == "fpiao":
            fpiao_effao = getattr(iao_mod, 'fpiao_effao', None)
            if ref_bas == "nao" and fpiao_effao is not None:
                U_nonorth, pmol = fpiao_effao(mol, get_c_src(True), x=local_w, mode='nao', pol_basis='ano', heavy_only=local_heavy_only, full_basis=False, mf=current_mf)
            else:
                U_nonorth, pmol = fpiao(mol, get_c_src(True), x=local_w, source_basis=ref_bas, pol_basis='ano', heavy_only=local_heavy_only, full_basis=False)
        elif p_base == "peiao":
            peiao_func = getattr(iao_mod, 'peiao', None)
            if peiao_func is not None:
                actual_mode = 'nao' if ref_bas == 'minao' else ref_bas
                U_nonorth, pmol = peiao_func(mol, get_c_src(True), mode=actual_mode, heavy_only=local_heavy_only, full_basis=False, mf=current_mf, x=local_w)
            else:
                raise NameError("PEIAO function not found in iao module")
        elif p_base == "iao":
            if ref_bas == "nao":
                U_nonorth, pmol = effao(mol, get_c_src(False), mode='nao', polarized=False, heavy_only=local_heavy_only, full_basis=False, mf=current_mf)
            else:
                U_nonorth, pmol = iao(mol, get_c_src(False), source_basis=ref_bas, heavy_only=local_heavy_only, full_basis=False)
        elif p_base == "iao-autosad":
            U_nonorth, pmol = autosad(mol, get_c_src(False), polarized=False, heavy_only=local_heavy_only, full_basis=False, mf=current_mf)
        elif p_base.startswith("iao-effao"):
            mode = p_base.replace("iao-effao-", "").replace("iao-effao", "net")
            if mode == "symmetric": mode = "sym"
            U_nonorth, pmol = effao(mol, get_c_src(False), mode=mode, polarized=False, heavy_only=local_heavy_only, full_basis=False, mf=current_mf)
        elif p_base == "wiao":
            U_nonorth, pmol = wiao(mol, get_c_src(False), heavy_only=local_heavy_only, full_basis=False)
        elif p_base == "iao-pyscf":
             from pyscf.lo import iao as pyscf_iao
             from pyscf.lo import orth
             C_iao_nonorth = pyscf_iao.iao(mol, get_c_src(False))
             U_nonorth = orth.vec_lowdin(C_iao_nonorth, S)
             pmol = mol
        else:
            raise NameError(f"Unknown IAO type: {p_base}")
        
        U = np.dot(S, U_nonorth)
        from esipy.tools import build_eta
        eta = build_eta(pmol)
        proj_c = c_full if c_full is not None else c
        return [np.linalg.multi_dot((proj_c.T, U, eta[i], U.T, proj_c)) for i in range(mol.natm)]

    # 1. UNRESTRICTED (check if actually unrestricted coefficients)
    is_unrest = isinstance(mf, scf.uhf.UHF) or (hasattr(mf, "__name__") and mf.__name__ == "UHF")
    if is_unrest and isinstance(mf.mo_coeff, (list, tuple)) and len(mf.mo_coeff) == 2:
        ca, cb = mf.mo_coeff
        oa, ob = np.asarray(mf.mo_occ[0]), np.asarray(mf.mo_occ[1])
        is_natorb = np.any((oa > 1e-6) & (np.abs(oa - 1.0) > 1e-6)) or np.any((ob > 1e-6) & (np.abs(ob - 1.0) > 1e-6))
        
        if is_natorb:
            mask_a, mask_b = oa > 1e-10, ob > 1e-10
            from esipy.iao import reference_mol
            pmol = reference_mol(mol, source_basis="minao")
            n_ref = pmol.nao
            aom_alpha = get_iao_aoms(partition_label, ca[:, :n_ref], mf, c_full=ca[:, mask_a])
            aom_beta = get_iao_aoms(partition_label, cb[:, :n_ref], mf, c_full=cb[:, mask_b])
            aom = [[aom_alpha, aom_beta], [np.diag(oa[mask_a]), np.diag(ob[mask_b])]]
        else:
            coeff_alpha = ca[:, oa > 0.5]
            coeff_beta = cb[:, ob > 0.5]
            aom = [get_iao_aoms(partition_label, coeff_alpha, mf), get_iao_aoms(partition_label, coeff_beta, mf)]
        
        if save: save_file(aom, save)
        return aom

    # 2. RESTRICTED (and potentially Natural Orbitals)
    else:
        occ = np.asarray(mf.mo_occ)
        is_natorb = np.any((occ > 1e-6) & (np.abs(occ - 1.0) > 1e-6) & (np.abs(occ - 2.0) > 1e-6))
        
        # If mf.mo_coeff was erroneously packed in UHF but actually restricted
        coeff_src = mf.mo_coeff[0] if isinstance(mf.mo_coeff, (list, tuple)) else mf.mo_coeff

        if is_natorb:
            mask = occ > 1e-10
            coeff = coeff_src[:, mask]
            occ_mask = occ[mask]
            from esipy.iao import reference_mol
            pmol = reference_mol(mol, source_basis="minao")
            n_ref = pmol.nao
            aom = get_iao_aoms(partition_label, coeff_src[:, :n_ref], mf, c_full=coeff)
            aom = [aom, np.diag(occ_mask)]
        else:
            mask = occ > 0.5
            coeff = coeff_src[:, mask]
            if partition_label in ("lowdin", "meta_lowdin", "nao", "mulliken"):
                aom = []
                if partition_label == "lowdin": U_inv = lowdin(S)
                elif partition_label == "meta_lowdin":
                    from pyscf.lo import orth
                    U_inv = orth.orth_ao(mf, method="meta_lowdin")
                elif partition_label == "nao": U_inv = nao.nao(mol, mf, S)
                
                if partition_label == "mulliken":
                    eta = build_eta(mol)
                    for i in range(mol.natm): aom.append(np.linalg.multi_dot((coeff.T, S, eta[i], coeff)))
                else:
                    U = np.linalg.inv(U_inv); eta = build_eta(mol)
                    for i in range(mol.natm): aom.append(coeff.T @ U.T @ eta[i] @ U @ coeff)
            else:
                aom = get_iao_aoms(partition_label, coeff, mf)
            
        if save: save_file(aom, save)
        return aom
