import numpy as np
import re
from pyscf.lo import nao
from pyscf.lo.orth import lowdin, orth_ao
from pyscf import scf

from esipy.tools import save_file, format_partition, get_natorbs, build_eta

def make_aoms(mol, mf, partition, myhf=None, save=None, iaomix=0.5, iaoref='minao', iaopol=None, heavy_only=False, full_basis=False):
    """
    Build the Atomic Overlap Matrices (AOMs) in the Molecular Orbitals basis.
    """

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
    S = mol.intor("int1e_ovlp") 

    # ADDED c_transform to decouple basis generation from orbital transformation
    def get_iao_aoms(p_type, c, current_mf, w_override=None, c_transform=None):
        if c_transform is None:
            c_transform = c

        try:
            import iao_dump as iao_mod
        except ImportError:
            import esipy.iao as iao_mod

        iao, fpiao, dfpiao = iao_mod.iao, iao_mod.fpiao, iao_mod.dfpiao
        get_effaos, reference_mol, autosad = iao_mod.get_effaos, iao_mod.reference_mol, iao_mod.autosad

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

        p_parts = p_type_clean.split()
        p_base = p_parts[0]
        ref_bas = p_parts[1] if len(p_parts) > 1 else iaoref

        if w_override is not None:
            local_w = w_override
        elif string_w is not None:
            local_w = string_w
        elif p_base == "fpiao":
            local_w = 1.0
        elif p_base == "dfpiao":
            local_w = 0.5
        else:
            local_w = weight

        if p_base == "dfpiao":
            # Pass c_transform down the recursive calls
            aom_iao = get_iao_aoms(f"iao {ref_bas}", c, current_mf, w_override=0.0, c_transform=c_transform)
            aom_fpiao = get_iao_aoms(f"fpiao {ref_bas}", c, current_mf, w_override=1.0, c_transform=c_transform)
            return [local_w * aom_iao[i] + (1 - local_w) * aom_fpiao[i] for i in range(mol.natm)]

        elif p_base in ["iao", "fpiao"]:
            if p_base == "iao":
                C_iao_orth, pmol = iao(mol, c, source_basis=ref_bas, heavy_only=heavy_only, full_basis=full_basis)
            else:
                # Pass actual heavy_only to respect $HPOL
                C_iao_orth, pmol = fpiao(mol, c, x=local_w, source_basis=ref_bas, pol_basis=iaopol if iaopol else "ano", heavy_only=heavy_only)

            S1 = mol.intor("int1e_ovlp")
            # Apply transformation to c_transform, not c
            C_mo_iao = C_iao_orth.T @ S1 @ c_transform
            aoslices = pmol.aoslice_by_atom()
            aoms = []
            for i in range(pmol.natm):
                s0, s1 = aoslices[i, 2], aoslices[i, 3]
                C_atom = C_mo_iao[s0:s1, :]
                aoms.append(C_atom.T @ C_atom)
            return aoms

        elif p_base.startswith("iao-effao-") or p_base in ["sym", "sps", "spsa", "iao-autosad"]:
            if p_base == "iao-autosad":
                U_nonorth, pmol = autosad(mol, current_mf, free_atom=True, heavy_only=heavy_only, full_basis=full_basis)
            else:
                mode = p_base.replace("iao-effao-", "")
                if mode == "symmetric": mode = "sym"
                U_nonorth, pmol = autosad(mol, current_mf, free_atom=False, mode=mode, heavy_only=heavy_only, full_basis=full_basis)
            if isinstance(U_nonorth, tuple):
                U_nonorth = U_nonorth[0] if np.allclose(c, current_mf.mo_coeff[0][:, :c.shape[1]]) else U_nonorth[1]
            U = np.dot(S, U_nonorth)
            eta = build_eta(pmol)
            # Apply transformation to c_transform
            return [np.linalg.multi_dot((c_transform.T, U, eta[i], U.T, c_transform)) for i in range(pmol.natm)]

        elif p_base == "iao_basis":
            import population_analysis
            p = current_mf.make_rdm1(ao_repr=True)
            if p.ndim == 3: p = p[0] + p[1]
            pop, bo, di = population_analysis.compute_iao_basis_partition(
                p, S, c_transform.flatten(), c_transform.shape[1], mol.atom_powers(), mol.atom_coords(),
                mol._bas[:, gto.ANG_OF], mol._bas[:, gto.ATOM_OF], mol._bas[:, gto.NPRIM_OF],
                mol.ao_loc_nr(), mol._env, mol._bas, mol.natm, True
            )
            return None

        elif p_base == "iao-pyscf":
             from pyscf.lo import iao as pyscf_iao
             from pyscf.lo import orth
             C_iao_nonorth = pyscf_iao.iao(mol, c)
             U_nonorth = orth.vec_lowdin(C_iao_nonorth, S)
             U = np.dot(S, U_nonorth)
             eta = build_eta(mol)
             # Apply transformation to c_transform
             return [np.linalg.multi_dot((c_transform.T, U, eta[i], U.T, c_transform)) for i in range(mol.natm)]

        else:
            raise NameError(f"Unknown IAO type: {p_type}")

    # ==========================================
    # 1. UNRESTRICTED
    # ==========================================
    if isinstance(mf, scf.uhf.UHF):
        coeff_alpha = mf.mo_coeff[0][:, mf.mo_occ[0] > 0]
        coeff_beta = mf.mo_coeff[1][:, mf.mo_occ[1] > 0]

        aom_alpha, aom_beta = [], []
        if partition in ("lowdin", "meta-lowdin", "nao"):
            if partition == "lowdin":
                U_inv = lowdin(S)
            elif partition == "meta-lowdin":
                U_inv = orth_ao(mf, "meta-lowdin", pre_orth_ao="ANO")
            elif partition == "nao":
                U_inv = nao.nao(mol, mf, S)
            U = np.linalg.inv(U_inv)
            eta = build_eta(mol)
            for i in range(mol.natm):
                aom_alpha.append(coeff_alpha.T @ U.T @ eta[i] @ U @ coeff_alpha)
                aom_beta.append(coeff_beta.T @ U.T @ eta[i] @ U @ coeff_beta)

        elif partition.startswith("iao") or partition in ["fpiao", "dfpiao"]:
            aom_alpha = get_iao_aoms(partition, coeff_alpha, mf)
            aom_beta = get_iao_aoms(partition, coeff_beta, mf)

        elif partition == "mulliken":
            eta = build_eta(mol)
            for i in range(mol.natm):
                aom_alpha.append(np.linalg.multi_dot((coeff_alpha.T, S, eta[i], coeff_alpha)))
                aom_beta.append(np.linalg.multi_dot((coeff_beta.T, S, eta[i], coeff_beta)))
        else:
            raise NameError(f"Hilbert-space scheme not available: {partition}")

        return [aom_alpha, aom_beta]

    # ==========================================
    # 2. RESTRICTED
    # ==========================================
    elif isinstance(mf, scf.hf.RHF):
        coeff = mf.mo_coeff[:, mf.mo_occ > 0]

        if partition in ("lowdin", "meta-lowdin", "nao"):
            if partition == "lowdin":
                U_inv = lowdin(S)
            elif partition == "meta-lowdin":
                U_inv = orth_ao(mf, "meta-lowdin", pre_orth_ao="ANO")
            elif partition == "nao":
                U_inv = nao.nao(mol, mf, S)
            U = np.linalg.inv(U_inv)
            eta = build_eta(mol)
            aom = []
            for i in range(mol.natm):
                aom.append(coeff.T @ U.T @ eta[i] @ U @ coeff)

        elif partition.startswith("iao") or partition in ["fpiao", "dfpiao"]:
            aom = get_iao_aoms(partition, coeff, mf)

        elif partition == "mulliken":
            eta = build_eta(mol)
            aom = []
            for i in range(mol.natm):
                aom.append(np.linalg.multi_dot((coeff.T, S, eta[i], coeff)))
        else:
            raise NameError(f"Hilbert-space scheme not available: {partition}")

        return aom

    # ==========================================
    # 3. POST-HF / MULTICONFIGURATIONAL
    # ==========================================
    else:
        S = mol.intor("int1e_ovlp")
        if "fci" in mf.__module__:
            raise NameError(" | FCI not supported yet")
        else:
            occ, coeff = get_natorbs(mf, S)

        aom = []
        if partition in ("lowdin", "meta-lowdin", "nao"):
            if partition == "lowdin":
                U_inv = lowdin(S)
            elif partition == "meta-lowdin":
                U_inv = orth_ao(mf, "meta-lowdin", pre_orth_ao="ANO")
            elif partition == "nao":
                if myhf is None:
                    raise NameError(" | Could not calculate partition from Natural Orbitals calculation \n | Please provide HF reference object in 'myhf'")
                U_inv = nao.nao(mol, myhf, S)
            U = np.linalg.inv(U_inv)
            eta = build_eta(mol)
            for i in range(mol.natm):
                SCR = coeff.T @ U.T @ eta[i]
                aom.append(SCR @ SCR.T)

        elif partition.startswith("iao") or partition in ["fpiao", "dfpiao"]:
            if myhf is None:
                raise NameError(" | Could not calculate partition from Natural Orbitals calculation \n | Please provide HF reference object in 'myhf'")

            # Extract HF coefficients to build the IAO projection
            coeff_hf = myhf.mo_coeff[:, myhf.mo_occ > 0]

            # Build using HF state, but transform the Natural Orbitals (coeff)
            aom = get_iao_aoms(partition, coeff_hf, myhf, c_transform=coeff)

        elif partition == "mulliken":
            eta = build_eta(mol)
            for i in range(mol.natm):
                SCR = np.linalg.multi_dot((coeff.T, S, eta[i], coeff))
                aom.append(SCR)
        else:
            raise NameError(f"Hilbert-space scheme not available: {partition}")

        if save:
            save_file([aom, occ], save)

        return [aom, occ]

