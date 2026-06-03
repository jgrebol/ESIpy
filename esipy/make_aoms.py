import numpy as np
import re
from pyscf.lo import nao
from pyscf.lo.orth import lowdin
from pyscf import scf

from esipy.tools import save_file, format_partition, get_natorbs, build_eta

def make_aoms(mol, mf, partition, myhf=None, save=None, is_fchk=False):
    """
    Build the Atomic Overlap Matrices (AOMs) in the Molecular Orbitals basis. 
    """

    partition_label = format_partition(partition)
    iaoref = 'sto-3g' if partition_label == 'iao2' else 'minao'
    
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

        m_w = re.search(r"\(+([\d\.]+)\)+", p_type)
        if m_w:
            string_w = float(m_w.group(1))
            p_type_clean = re.sub(r"\(+[\d\.]+\)+", "", p_type).replace("  ", " ").strip()
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
        if p_base in ["iao", "iao-autosad", "fpiao", "dfpiao", "wiao"]:
            iaoref = 'minao'
        elif p_base in ["iao2", "iao-effao", "iao-autosad2", "peiao", "dpeiao", "fpiao2", "dfpiao2"] or p_base.startswith("iao-effao"):
            iaoref = 'valence'
        else:
            iaoref = 'minao'

        ref_bas = p_parts[1] if len(p_parts) > 1 else iaoref
        weight = 0.5
        local_w = w_override if w_override is not None else (string_w if string_w is not None else (weight if weight is not None else 0.5))

        # IAO construction for Natural Orbitals:
        # use first N_ref NOs for construction, but project all NOs
        is_no = False
        if hasattr(current_mf, 'mo_occ') and getattr(current_mf, 'mo_occ', None) is not None:
            occ_tmp = np.asarray(current_mf.mo_occ)
            if np.any((occ_tmp > 1e-6) & (np.abs(occ_tmp - 1.0) > 1e-6) & (np.abs(occ_tmp - 2.0) > 1e-6)):
                is_no = True

        def get_c_src(is_pol):
            src = c
            if not is_no: return src
            pmol_ref = reference_mol(mol, polarized=is_pol, source_basis=ref_bas, heavy_only=local_heavy_only, full_basis=False)
            return src[:, :pmol_ref.nao]

        if p_base == "dfpiao":
            aom_iao = get_iao_aoms(f"iao {ref_bas}", c, current_mf, current_myhf=current_myhf, c_full=c_full)
            aom_fpiao = get_iao_aoms(f"fpiao {ref_bas}", c, current_mf, w_override=1.0, current_myhf=current_myhf, c_full=c_full)
            return [local_w * aom_iao[i] + (1.0 - local_w) * aom_fpiao[i] for i in range(mol.natm)]
        elif p_base == "dfpiao2":
            aom_iao = get_iao_aoms(f"iao2 {ref_bas}", c, current_mf, current_myhf=current_myhf, c_full=c_full)
            aom_fpiao = get_iao_aoms(f"fpiao2 {ref_bas}", c, current_mf, w_override=1.0, current_myhf=current_myhf, c_full=c_full)
            return [local_w * aom_iao[i] + (1.0 - local_w) * aom_fpiao[i] for i in range(mol.natm)]
        elif p_base == "dpeiao":
            actual_mode = 'nao' if ref_bas in ['minao', 'valence'] else ref_bas
            aom_iao = get_iao_aoms(f"iao-effao-{actual_mode}", c, current_mf, current_myhf=current_myhf, c_full=c_full)
            aom_peiao = get_iao_aoms(f"peiao {ref_bas}", c, current_mf, current_myhf=current_myhf, c_full=c_full)
            return [local_w * aom_iao[i] + (1.0 - local_w) * aom_peiao[i] for i in range(mol.natm)]

        elif p_base in ["fpiao", "fpiao2"]:
            fpiao_effao = getattr(iao_mod, 'fpiao_effao', None)
            if ref_bas == "nao" and fpiao_effao is not None:
                U_nonorth, pmol = fpiao_effao(mol, get_c_src(True), x=local_w, mode='nao', pol_basis='ano', heavy_only=local_heavy_only, full_basis=False, mf=current_mf)
            else:
                U_nonorth, pmol = fpiao(mol, get_c_src(True), x=local_w, source_basis=ref_bas, pol_basis='ano', heavy_only=local_heavy_only, full_basis=False)
        elif p_base == "peiao":
            peiao_func = getattr(iao_mod, 'peiao', None)
            if peiao_func is not None:
                actual_mode = 'nao' if ref_bas in ['minao', 'valence'] else ref_bas
                U_nonorth, pmol = peiao_func(mol, get_c_src(True), mode=actual_mode, heavy_only=local_heavy_only, full_basis=False, mf=current_mf, x=local_w)
            else:
                raise NameError("PEIAO function not found in iao module")
        elif p_base in ["iao", "iao2"]:
            if ref_bas == "nao":
                U_nonorth, pmol = effao(mol, get_c_src(False), mode='nao', polarized=False, heavy_only=local_heavy_only, full_basis=False, mf=current_mf)
            else:
                U_nonorth, pmol = iao(mol, get_c_src(False), source_basis=ref_bas, heavy_only=local_heavy_only, full_basis=False)
        elif p_base in ["iao-autosad", "iao-autosad2"]:
            U_nonorth, pmol = autosad(mol, get_c_src(False), polarized=False, heavy_only=local_heavy_only, full_basis=False, mf=current_mf, source_basis=ref_bas)
        elif p_base.startswith("iao-effao"):
            mode = p_base.replace("iao-effao-", "").replace("iao-effao", "net")
            if mode == "symmetric": mode = "sym"
            U_nonorth, pmol = effao(mol, get_c_src(False), mode=mode, polarized=False, heavy_only=local_heavy_only, full_basis=False, mf=current_mf)
        elif p_base == "wiao":
            U_nonorth, pmol = wiao(mol, get_c_src(False), heavy_only=local_heavy_only, full_basis=False)
        elif p_base == "iao-pyscf":
             from pyscf.lo import iao as pyscf_iao
             from pyscf.lo import orth
             try:
                 from esipy.iao import reference_mol
             except ImportError:
                 from iao import reference_mol
             pmol = reference_mol(mol)
             C_iao_nonorth = pyscf_iao.iao(mol, get_c_src(False))
             U_nonorth = orth.vec_lowdin(C_iao_nonorth, S)
        else:
            raise NameError(f"Unknown IAO type: {p_base}")

        U = np.dot(S, U_nonorth)
        from esipy.tools import build_eta
        eta = build_eta(pmol)
        proj_c = c_full if c_full is not None else c
        return [np.linalg.multi_dot((proj_c.T, U, eta[i], U.T, proj_c)) for i in range(mol.natm)]

    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    if mo_occ is None and hasattr(mf, '_scf'):
        mo_occ = mf._scf.mo_occ
        

    is_unrest = (isinstance(mo_coeff, (list, tuple)) and len(mo_coeff) == 2) or \
                (isinstance(mo_coeff, np.ndarray) and mo_coeff.ndim == 3 and mo_coeff.shape[0] == 2)
    
    from esipy.tools import is_natorb_wf
    
    # 1. UNRESTRICTED
    if is_unrest:
        is_natorb = False
        if is_fchk:
            d_lbl = getattr(mf, 'density_label', '')
            if 'MP2' in d_lbl or 'CC' in d_lbl or 'CI' in d_lbl:
                is_natorb = True
            else:
                if mo_occ is not None and np.asarray(mo_occ).dtype != object:
                    occ_flat = np.asarray(mo_occ).flatten()
                    is_natorb = np.any((occ_flat > 1e-6) & (np.abs(occ_flat - 1.0) > 1e-6) & (np.abs(occ_flat - 2.0) > 1e-6))
        else:
            if is_natorb_wf(mf):
                is_natorb = True

        if is_natorb:
            occ, coeff = get_natorbs(mf, S)
            if isinstance(occ, list):
                oa, ob = np.asarray(occ[0]), np.asarray(occ[1])
                ca, cb = coeff[0], coeff[1]
            else:
                oa, ob = np.asarray(occ), np.asarray(occ)
                ca, cb = coeff, coeff
            mask_a = np.ones(len(oa), dtype=bool)
            mask_b = np.ones(len(ob), dtype=bool)
        else:
            ca, cb = mo_coeff[0], mo_coeff[1]
            oa, ob = np.asarray(mo_occ[0]), np.asarray(mo_occ[1])
            mask_a = oa > 0.5
            mask_b = ob > 0.5
            
        if partition_label in ("lowdin", "meta-lowdin", "nao", "mulliken"):
            aom_alpha, aom_beta = [], []
            if partition_label == "lowdin": U_inv = lowdin(S)
            elif partition_label == "meta-lowdin":
                from pyscf.lo import orth
                from esipy.tools import RefUHF
                ref_mf = RefUHF(mol, [ca, cb], [oa, ob]) if is_natorb else (myhf if myhf is not None else mf)
                U_inv = orth.orth_ao(ref_mf, method="meta-lowdin")
            elif partition_label == "nao":
                from esipy.tools import RefUHF
                ref_mf = RefUHF(mol, [ca, cb], [oa, ob]) if is_natorb else mf
                U_inv = nao.nao(mol, ref_mf, S)
            
            if partition_label == "mulliken":
                eta = build_eta(mol)
                for i in range(mol.natm):
                    aom_alpha.append(np.linalg.multi_dot((ca[:, mask_a].T, S, eta[i], ca[:, mask_a])))
                    aom_beta.append(np.linalg.multi_dot((cb[:, mask_b].T, S, eta[i], cb[:, mask_b])))
            else:
                U = np.linalg.inv(U_inv); eta = build_eta(mol)
                for i in range(mol.natm):
                    aom_alpha.append(ca[:, mask_a].T @ U.T @ eta[i] @ U @ ca[:, mask_a])
                    aom_beta.append(cb[:, mask_b].T @ U.T @ eta[i] @ U @ cb[:, mask_b])
            
            if is_natorb and (not is_fchk or np.any(np.abs(oa - 1.0) > 1e-6)):
                aom = [[aom_alpha, aom_beta], [oa[mask_a], ob[mask_b]]]
            else:
                aom = [aom_alpha, aom_beta]
        else:
            # IAO logic
            if is_natorb:
                from esipy.iao import reference_mol
                from esipy.tools import RefUHF, RefRHF
                pmol = reference_mol(mol, source_basis="minao")
                
                if hasattr(mf, 'ncas') and getattr(mf, 'ncas') is not None:
                    n_occ_act = getattr(mf, 'ncore', 0) + getattr(mf, 'ncas', 0)
                    ca_trunc = ca[:, :n_occ_act]
                    cb_trunc = cb[:, :n_occ_act]
                else:
                    ca_trunc = ca[:, :pmol.nao]
                    cb_trunc = cb[:, :pmol.nao]
                
                ref_mfa = RefRHF(mol, ca, oa)
                ref_mfb = RefRHF(mol, cb, ob)
                aom_alpha = get_iao_aoms(partition_label, ca_trunc, ref_mfa, c_full=ca[:, mask_a])
                aom_beta = get_iao_aoms(partition_label, cb_trunc, ref_mfb, c_full=cb[:, mask_b])
                aom = [[aom_alpha, aom_beta], [oa[mask_a], ob[mask_b]]]
            else:
                aom_alpha = get_iao_aoms(partition_label, ca[:, mask_a], mf)
                aom_beta = get_iao_aoms(partition_label, cb[:, mask_b], mf)
                aom = [aom_alpha, aom_beta]
        
        if save: save_file(aom, save)
        return aom

    # 2. RESTRICTED
    else:
        mo_coeff = mf.mo_coeff
        mo_occ = mf.mo_occ
        if mo_occ is None and hasattr(mf, '_scf'):
            mo_occ = mf._scf.mo_occ
            
        if isinstance(mo_coeff, (list, tuple)) and len(mo_coeff) == 1:
            mo_coeff = mo_coeff[0]
            mo_occ = mo_occ[0]
            
        is_natorb = False
        if is_fchk:
            d_lbl = getattr(mf, 'density_label', '')
            if 'MP2' in d_lbl or 'CC' in d_lbl or 'CI' in d_lbl:
                is_natorb = True
            else:
                if mo_occ is not None and np.asarray(mo_occ).dtype != object:
                    occ_flat = np.asarray(mo_occ).flatten()
                    is_natorb = np.any((occ_flat > 1e-6) & (np.abs(occ_flat - 1.0) > 1e-6) & (np.abs(occ_flat - 2.0) > 1e-6))
        else:
            if is_natorb_wf(mf):
                is_natorb = True
                
        if is_natorb:
            occ, mo_coeff = get_natorbs(mf, S)
            if hasattr(mf, 'ncas') and getattr(mf, 'ncas') is not None:
                n_occ_act = getattr(mf, 'ncore', 0) + getattr(mf, 'ncas', 0)
                mask = np.zeros(len(occ), dtype=bool)
                mask[:n_occ_act] = True
            else:
                mask = np.ones(len(occ), dtype=bool) # DO NOT MASK NATURAL ORBITALS
        else:
            occ = np.asarray(mo_occ)
            mask = (occ > 0.5)
            
        coeff_src = mo_coeff
        coeff = coeff_src[:, mask]

        if partition_label in ("lowdin", "meta-lowdin", "nao", "mulliken"):
            aom = []
            if partition_label == "lowdin": U_inv = lowdin(S)
            elif partition_label == "meta-lowdin":
                from pyscf.lo import orth
                from esipy.tools import RefRHF
                ref_mf = RefRHF(mol, coeff_src, occ) if is_natorb else (myhf if myhf is not None else mf)
                U_inv = orth.orth_ao(ref_mf, method="meta-lowdin")
            elif partition_label == "nao":
                from esipy.tools import RefRHF
                ref_mf = RefRHF(mol, coeff_src, occ) if is_natorb else mf
                U_inv = nao.nao(mol, ref_mf, S)
            
            if partition_label == "mulliken":
                eta = build_eta(mol)
                for i in range(mol.natm): aom.append(np.linalg.multi_dot((coeff.T, S, eta[i], coeff)))
            else:
                U = np.linalg.inv(U_inv); eta = build_eta(mol)
                for i in range(mol.natm): aom.append(coeff.T @ U.T @ eta[i] @ U @ coeff)
            
            if is_natorb:
                aom = [aom, occ[mask]]
        else:
            # IAO logic
            if is_natorb:
                from esipy.iao import reference_mol
                from esipy.tools import RefRHF
                pmol = reference_mol(mol, source_basis="minao")
                
                if hasattr(mf, 'ncas') and getattr(mf, 'ncas') is not None:
                    n_occ_act = getattr(mf, 'ncore', 0) + getattr(mf, 'ncas', 0)
                    c_trunc = coeff_src[:, :n_occ_act]
                else:
                    c_trunc = coeff_src[:, :pmol.nao]
                
                ref_mf = RefRHF(mol, coeff_src, occ)
                aom = get_iao_aoms(partition_label, c_trunc, ref_mf, c_full=coeff)
                aom = [aom, occ[mask]]
            else:
                aom = get_iao_aoms(partition_label, coeff, mf)
                
        if save: save_file(aom, save)
        return aom
