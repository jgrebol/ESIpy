import numpy as np
import re
from pyscf.lo import nao
from pyscf.lo.orth import lowdin
from pyscf import scf

from esipy.tools import save_file, format_partition, get_natorbs, build_eta

def make_aoms(mol, mf, partition, myhf=None, save=None, iaomix=0.5):
    """
    Build the Atomic Overlap Matrices (AOMs) in the Molecular Orbitals basis.
    """

    # Parse the weight if present in partition string like piao-iao(0.5)
    weight = iaomix
    if isinstance(weight, list):
        weight = weight[0]
    match = re.search(r"\((.*?)\)", partition)
    if match:
        try:
            weight = float(match.group(1))
            partition = partition.split("(")[0].strip().lower()
        except:
            pass
    
    partition = partition.lower()
    # Check for IAO-ANO or PIAO-ANO and remove it
    if "ano" in partition:
        partition = partition.replace("-ano", "")

    S = mf.get_ovlp()
    
    def get_iao_aoms(p_type, c, current_mf, w_override=None):
        from esipy.iao import iao, piao, fpiao, get_effaos, reference_mol, autosad
        
        w = w_override if w_override is not None else weight

        if p_type.startswith("iao_"):
            basis = p_type.split("_")[1]
            U_nonorth, pmol = iao(mol, c, source_basis=basis)
        elif p_type == "iao":
            U_nonorth, pmol = iao(mol, c)
        elif p_type == "piao":
            U_nonorth, pmol = piao(mol, c)
        elif p_type in ["fpiao", "fpiao1"]:
            U_nonorth, pmol = fpiao(mol, c, x=w)
        elif p_type == "dfpiao":
            # DFPIAO = w * IAO + (1-w) * FPIAO
            aom_iao = get_iao_aoms("iao", c, current_mf)
            aom_fpiao = get_iao_aoms("fpiao", c, current_mf, w_override=1.0)
            pmol = reference_mol(mol, polarized=False) 
            return [w * aom_iao[i] + (1 - w) * aom_fpiao[i] for i in range(pmol.natm)]
        elif p_type == "xiao_dfpiao":
            aom_iao = get_iao_aoms("iao", c, current_mf)
            aom_fpiao = get_iao_aoms("fpiao", c, current_mf, w_override=1.0)
            pmol = reference_mol(mol, polarized=False)
            return [(1 - w) * aom_iao[i] + w * aom_fpiao[i] for i in range(pmol.natm)]
        elif p_type.startswith("iao-effao-") or p_type in ["sym", "sps", "spsa", "iao-autosad"]:
            if p_type == "iao-autosad":
                U_nonorth, pmol = autosad(mol, current_mf, free_atom=True)
            else:
                mode = p_type.replace("iao-effao-", "")
                if mode == "symmetric": mode = "sym"
                U_nonorth, pmol = autosad(mol, current_mf, free_atom=False, mode=mode)
            
            # Handle unrestricted return (tuple)
            if isinstance(U_nonorth, tuple):
                # This part is tricky because get_iao_aoms is called per spin.
                # But autosad returns (Ua, Ub) if mf is UHF.
                # current_mf might be mf or myhf.
                # If we are here, we should check which spin we want.
                # Actually, make_aoms calls get_iao_aoms with coeff_alpha/beta.
                # We need a way to pass the right spin to autosad or get the right spin from it.
                # Simplified: if current_mf is UHF, we expect c to be either ca or cb.
                # We can check if c is ca or cb.
                if np.allclose(c, current_mf.mo_coeff[0][:, :c.shape[1]]):
                    U_nonorth = U_nonorth[0]
                else:
                    U_nonorth = U_nonorth[1]
        else:
            raise NameError(f"Unknown IAO type: {p_type}")
        
        # Now U_nonorth is the orthonormalized IAO transformation
        # For IAO, the transformation matrix U is S^-1 T? No.
        # orth.vec_lowdin returns T such that T.T S T = I.
        # The AOM is C.T S T eta T.T S C.
        # This matches the logic below.
        U = np.dot(S, U_nonorth)
        eta = build_eta(pmol)
        return [np.linalg.multi_dot((c.T, U, eta[i], U.T, c)) for i in range(pmol.natm)]

    # UNRESTRICTED
    if isinstance(mf, scf.uhf.UHF):
        nocc_alpha = mf.mo_occ[0].astype(int)
        nocc_beta = mf.mo_occ[1].astype(int)
        coeff_alpha = mf.mo_coeff[0][:, : nocc_alpha.sum()]
        coeff_beta = mf.mo_coeff[1][:, : nocc_beta.sum()]

        aom_alpha, aom_beta = [], []
        if partition in ("lowdin", "meta_lowdin", "nao"):
            if partition == "lowdin":
                U_inv = lowdin(S)
            elif partition == "meta_lowdin":
                from pyscf.lo.orth import restore_ao_character
                pre_orth_ao = restore_ao_character(mol, "MINAO")
                w = np.ones(pre_orth_ao.shape[1])
                U_inv = nao._nao_sub(mol, w, pre_orth_ao, S)
            elif partition == "nao":
                U_inv = nao.nao(mol, mf, S)
            U = np.linalg.inv(U_inv)
            eta = build_eta(mol)
            for i in range(mol.natm):
                aom_alpha.append(coeff_alpha.T @ U.T @ eta[i] @ U @ coeff_alpha)
                aom_beta.append(coeff_beta.T @ U.T @ eta[i] @ U @ coeff_beta)

        elif partition.startswith("iao") or partition in ["piao", "fpiao", "fpiao1", "dfpiao", "xiao_dfpiao"] or partition == "piao-iao":
            if partition == "piao-iao":
                print(f" | Averaging AOMs for combined partition (IAO weight = {weight})...")
                aom_iao_a = get_iao_aoms("iao", coeff_alpha, mf)
                aom_piao_a = get_iao_aoms("piao", coeff_alpha, mf)
                aom_alpha = [weight * aom_iao_a[i] + (1 - weight) * aom_piao_a[i] for i in range(mol.natm)]
                
                aom_iao_b = get_iao_aoms("iao", coeff_beta, mf)
                aom_piao_b = get_iao_aoms("piao", coeff_beta, mf)
                aom_beta = [weight * aom_iao_b[i] + (1 - weight) * aom_piao_b[i] for i in range(mol.natm)]
            else:
                aom_alpha = get_iao_aoms(partition, coeff_alpha, mf)
                aom_beta = get_iao_aoms(partition, coeff_beta, mf)

        elif partition == "mulliken":
            eta = build_eta(mol)
            for i in range(mol.natm):
                aom_alpha.append(np.linalg.multi_dot((coeff_alpha.T, S, eta[i], coeff_alpha)))
                aom_beta.append(np.linalg.multi_dot((coeff_beta.T, S, eta[i], coeff_beta)))
        else:
            raise NameError(f"Hilbert-space scheme not available: {partition}")

        aom = [aom_alpha, aom_beta]

    # RESTRICTED
    elif isinstance(mf, scf.hf.RHF):
        coeff = mf.mo_coeff[:, mf.mo_occ > 0]
        aom = []
        if partition in ("lowdin", "meta_lowdin", "nao"):
            if partition == "lowdin":
                U_inv = lowdin(S)
            elif partition == "meta_lowdin":
                from pyscf.lo.orth import restore_ao_character
                pre_orth_ao = restore_ao_character(mol, "MINAO")
                w = np.ones(pre_orth_ao.shape[1])
                U_inv = nao._nao_sub(mol, w, pre_orth_ao, S)
            elif partition == "nao":
                U_inv = nao.nao(mol, mf, S)
            U = np.linalg.inv(U_inv)
            eta = build_eta(mol)
            for i in range(mol.natm):
                aom.append(coeff.T @ U.T @ eta[i] @ U @ coeff)

        elif partition.startswith("iao") or partition in ["piao", "fpiao", "fpiao1", "dfpiao", "xiao_dfpiao"] or partition == "piao-iao":
            if partition == "piao-iao":
                print(f" | Averaging AOMs for combined partition (IAO weight = {weight})...")
                aom_iao = get_iao_aoms("iao", coeff, mf)
                aom_piao = get_iao_aoms("piao", coeff, mf)
                aom = [weight * aom_iao[i] + (1 - weight) * aom_piao[i] for i in range(mol.natm)]
            else:
                aom = get_iao_aoms(partition, coeff, mf)

        elif partition == "mulliken":
            eta = build_eta(mol)
            for i in range(mol.natm):
                aom.append(np.linalg.multi_dot((coeff.T, S, eta[i], coeff)))
        else:
            raise NameError(f"Hilbert-space scheme not available: {partition}")

    # NATURAL ORBITALS
    else:
        if "fci" in mf.__module__:
            raise NameError(" | FCI not supported yet")
        else:
            occ, coeff = get_natorbs(mf, S)

        # HF instance required for IAO base
        if myhf is None and (partition.startswith("iao") or partition in ["piao", "fpiao", "fpiao1", "dfpiao", "xiao_dfpiao"] or partition == "piao-iao"):
             raise NameError(" | HF reference 'myhf' required for IAO-based partitions from Natural Orbitals")

        aom = []
        if partition in ("lowdin", "meta_lowdin", "nao"):
            if partition == "lowdin":
                U_inv = lowdin(S)
            elif partition == "meta_lowdin":
                from pyscf.lo.orth import restore_ao_character
                pre_orth_ao = restore_ao_character(mol, "MINAO")
                w = np.ones(pre_orth_ao.shape[1])
                U_inv = nao._nao_sub(mol, w, pre_orth_ao, S)
            elif partition == "nao":
                U_inv = nao.nao(mol, mf, S)
            U = np.linalg.inv(U_inv)
            eta = build_eta(mol)
            for i in range(mol.natm):
                aom.append(coeff.T @ U.T @ eta[i] @ U @ coeff)

        elif partition.startswith("iao") or partition in ["piao", "fpiao", "fpiao1", "dfpiao", "xiao_dfpiao"] or partition == "piao-iao":
            # For IAO, we use the MOs from the HF reference (myhf)
            coeff_hf = myhf.mo_coeff
            if coeff_hf.ndim == 3: coeff_hf = coeff_hf[0] # RHF part of UHF? simplified.
            
            if partition == "piao-iao":
                print(f" | Averaging AOMs for combined partition (IAO weight = {weight})...")
                from esipy.iao import iao, piao
                def get_U_iao(p_type, c_hf):
                    if p_type == "iao": U_nonorth, pmol = iao(mol, c_hf)
                    else: U_nonorth, pmol = piao(mol, c_hf)
                    U_inv = np.dot(U_nonorth, lowdin(np.linalg.multi_dot((U_nonorth.T, S, U_nonorth))))
                    return np.dot(S, U_inv), pmol

                U_iao, pmol_iao = get_U_iao("iao", coeff_hf)
                U_piao, pmol_piao = get_U_iao("piao", coeff_hf)
                eta = build_eta(pmol_iao) 
                
                for i in range(pmol_iao.natm):
                    aom_iao_i = np.linalg.multi_dot((coeff.T, U_iao, eta[i], U_iao.T, coeff))
                    aom_piao_i = np.linalg.multi_dot((coeff.T, U_piao, eta[i], U_piao.T, coeff))
                    aom.append(weight * aom_iao_i + (1 - weight) * aom_piao_i)
            else:
                aom = get_iao_aoms(partition, coeff, myhf) 

        elif partition == "mulliken":
            eta = build_eta(mol)
            for i in range(mol.natm):
                aom.append(np.linalg.multi_dot((coeff.T, S, eta[i], coeff)))
        else:
            raise NameError(f"Hilbert-space scheme not available: {partition}")

        aom = [aom, occ]

    if save: save_file(aom, save)
    return aom
