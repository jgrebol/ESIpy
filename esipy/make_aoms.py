import numpy as np
import re
from pyscf.lo import nao
from pyscf.lo.orth import lowdin
from pyscf import scf

from esipy.tools import save_file, format_partition, get_natorbs, build_eta

def make_aoms(mol, mf, partition, myhf=None, save=None, iaomix=None, iaoref='minao', iaopol='ano', heavy_only=None, full_basis=False):
    """
    Build the Atomic Overlap Matrices (AOMs) in the Molecular Orbitals basis. 
    """

    weight = iaomix
    if isinstance(weight, list):
        weight = weight[0]

    partition_label = format_partition(partition, iaomix=weight).lower()
    
    try:
        S = mf.get_ovlp()
    except:
        S = mol.get_ovlp()
    
    def get_iao_aoms(p_type, c, current_mf, w_override=None, current_myhf=None):
        try:
            import iao_dump as iao_mod
        except ImportError:
            import esipy.iao as iao_mod
        
        iao, fpiao, dfpiao = iao_mod.iao, iao_mod.fpiao, iao_mod.dfpiao
        effao, autosad, wiao = iao_mod.effao, iao_mod.autosad, iao_mod.wiao

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

        # HPOL handling
        local_heavy_only = heavy_only
        if "$hpol" in p_type_clean:
            local_heavy_only = False
            p_type_clean = p_type_clean.replace("$hpol", "").strip()
        
        # Defaults for heavy_only if not specified
        if local_heavy_only is None:
            if "fpiao" in p_type_clean or "dfpiao" in p_type_clean:
                local_heavy_only = True
            else:
                local_heavy_only = False

        p_parts = p_type_clean.split()
        p_base = p_parts[0]
        if p_base == "iao-basis":
            p_base = "iao"
            
        ref_bas = p_parts[1] if len(p_parts) > 1 else iaoref
        local_w = w_override if w_override is not None else (string_w if string_w is not None else weight)

        # Reference for IAO transformation
        mf_ref = current_myhf if current_myhf is not None else current_mf
        
        # Note: IAO variants now return a shared transformation matrix for UHF
        if p_base == "dfpiao":
            aom_iao = get_iao_aoms(f"iao {ref_bas}", c, current_mf, current_myhf=current_myhf)
            aom_fpiao = get_iao_aoms(f"fpiao {ref_bas}", c, current_mf, w_override=1.0, current_myhf=current_myhf)
            return [local_w * aom_iao[i] + (1.0 - local_w) * aom_fpiao[i] for i in range(mol.natm)]
        elif p_base == "xiao_dfpiao":
            aom_iao = get_iao_aoms(f"iao {ref_bas}", c, current_mf, current_myhf=current_myhf)
            aom_fpiao = get_iao_aoms(f"fpiao {ref_bas}", c, current_mf, w_override=1.0, current_myhf=current_myhf)
            return [(1.0 - local_w) * aom_iao[i] + local_w * aom_fpiao[i] for i in range(mol.natm)]
        elif p_base == "fpiao":
            U_nonorth, pmol = fpiao(mol, mf_ref, x=local_w, source_basis=ref_bas, pol_basis=iaopol, heavy_only=local_heavy_only, full_basis=full_basis)
        elif p_base == "iao":
            U_nonorth, pmol = iao(mol, mf_ref, source_basis=ref_bas, heavy_only=local_heavy_only, full_basis=full_basis)
        elif p_base == "iao-autosad":
            U_nonorth, pmol = autosad(mol, mf_ref, polarized=False, heavy_only=local_heavy_only, full_basis=full_basis)
        elif p_base.startswith("iao-effao"):
            mode = p_base.replace("iao-effao-", "").replace("iao-effao", "net")
            if mode == "symmetric": mode = "sym"
            U_nonorth, pmol = effao(mol, mf_ref, mode=mode, polarized=False, heavy_only=local_heavy_only, full_basis=full_basis)
        elif p_base == "wiao":
            U_nonorth, pmol = wiao(mol, mf_ref, heavy_only=local_heavy_only, full_basis=full_basis)
        elif p_base == "iao-pyscf":
             from pyscf.lo import iao as pyscf_iao
             from pyscf.lo import orth
             try:
                 import iao_dump as iao_mod
             except ImportError:
                 import esipy.iao as iao_mod
             c_occ = iao_mod.get_union_occ(mol, mf_ref)
             C_iao_nonorth = pyscf_iao.iao(mol, c_occ)
             U_nonorth = orth.vec_lowdin(C_iao_nonorth, S)
             pmol = mol
        else:
            raise NameError(f"Unknown IAO type: {p_base}")
        
        U = np.dot(S, U_nonorth)
        from esipy.tools import build_eta
        eta = build_eta(pmol)
        return [np.linalg.multi_dot((c.T, U, eta[i], U.T, c)) for i in range(mol.natm)]

    # 1. UNRESTRICTED
    if isinstance(mf, scf.uhf.UHF) or (hasattr(mf, "__name__") and mf.__name__ == "UHF"):
        coeff_alpha = mf.mo_coeff[0][:, mf.mo_occ[0] > 0]
        coeff_beta = mf.mo_coeff[1][:, mf.mo_occ[1] > 0]

        if partition_label in ("lowdin", "meta-lowdin", "metalowdin", "m-lowdin", "mlowdin", "nao", "mulliken"):
            aom_alpha, aom_beta = [], []
            if partition_label == "lowdin":
                U_inv = lowdin(S)
            elif partition_label in ("meta-lowdin", "metalowdin", "m-lowdin", "mlowdin"):
                from pyscf.lo.orth import restore_ao_character
                pre_orth_ao = restore_ao_character(mol, "MINAO")
                w = np.ones(pre_orth_ao.shape[1])
                U_inv = nao._nao_sub(mol, w, pre_orth_ao, S)
            elif partition_label == "nao":
                U_inv = nao.nao(mol, mf, S)
            
            if partition_label == "mulliken":
                eta = build_eta(mol)
                for i in range(mol.natm):
                    aom_alpha.append(np.linalg.multi_dot((coeff_alpha.T, S, eta[i], coeff_alpha)))
                    aom_beta.append(np.linalg.multi_dot((coeff_beta.T, S, eta[i], coeff_beta)))
            else:
                U = np.linalg.inv(U_inv)
                eta = build_eta(mol)
                for i in range(mol.natm):
                    aom_alpha.append(coeff_alpha.T @ U.T @ eta[i] @ U @ coeff_alpha)
                    aom_beta.append(coeff_beta.T @ U.T @ eta[i] @ U @ coeff_beta)
        else:
            aom_alpha = get_iao_aoms(partition_label, coeff_alpha, mf)
            aom_beta = get_iao_aoms(partition_label, coeff_beta, mf)
        
        aom = [aom_alpha, aom_beta]
        if save: save_file(aom, save)
        return aom

    # 2. RESTRICTED
    if isinstance(mf, scf.hf.RHF) or (hasattr(mf, "__name__") and mf.__name__ == "RHF"):
        coeff = mf.mo_coeff[:, mf.mo_occ > 0]

        if partition_label in ("lowdin", "meta-lowdin", "metalowdin", "m-lowdin", "mlowdin", "nao", "mulliken"):
            aom = []
            if partition_label == "lowdin":
                U_inv = lowdin(S)
            elif partition_label in ("meta-lowdin", "metalowdin", "m-lowdin", "mlowdin"):
                from pyscf.lo.orth import restore_ao_character
                pre_orth_ao = restore_ao_character(mol, "MINAO")
                w = np.ones(pre_orth_ao.shape[1])
                U_inv = nao._nao_sub(mol, w, pre_orth_ao, S)
            elif partition_label == "nao":
                U_inv = nao.nao(mol, mf, S)
            
            if partition_label == "mulliken":
                eta = build_eta(mol)
                for i in range(mol.natm):
                    aom.append(np.linalg.multi_dot((coeff.T, S, eta[i], coeff)))
            else:
                U = np.linalg.inv(U_inv)
                eta = build_eta(mol)
                for i in range(mol.natm):
                    aom.append(coeff.T @ U.T @ eta[i] @ U @ coeff)
        else:
            aom = get_iao_aoms(partition_label, coeff, mf)
            
        if save: save_file(aom, save)
        return aom

    # 3. NATURAL ORBITALS (Multireference / post-HF)
    else:
        occ, coeff = get_natorbs(mf, S)
        
        if partition_label in ("lowdin", "meta-lowdin", "metalowdin", "m-lowdin", "mlowdin", "nao", "mulliken"):
            aom = []
            if partition_label == "lowdin":
                U_inv = lowdin(S)
            elif partition_label in ("meta-lowdin", "metalowdin", "m-lowdin", "mlowdin"):
                from pyscf.lo.orth import restore_ao_character
                pre_orth_ao = restore_ao_character(mol, "MINAO")
                w = np.ones(pre_orth_ao.shape[1])
                U_inv = nao._nao_sub(mol, w, pre_orth_ao, S)
            elif partition_label == "nao":
                mf_ref = myhf if myhf is not None else mf
                U_inv = nao.nao(mol, mf_ref, S)
            
            if partition_label == "mulliken":
                eta = build_eta(mol)
                for i in range(mol.natm):
                    aom.append(np.linalg.multi_dot((coeff.T, S, eta[i], coeff)))
            else:
                U = np.linalg.inv(U_inv)
                eta = build_eta(mol)
                for i in range(mol.natm):
                    aom.append(coeff.T @ U.T @ eta[i] @ U @ coeff)
        else:
            aom = get_iao_aoms(partition_label, coeff, mf, current_myhf=myhf)

        aom_result = [aom, occ]
        if save: save_file(aom_result, save)
        return aom_result
