import numpy as np
import re
from pyscf.lo import nao
from pyscf.lo.orth import lowdin
from pyscf import scf

from esipy.tools import save_file, format_partition, get_natorbs, build_eta, build_iao_aoms

def make_aoms(mol, mf, partition, myhf=None, save=None, is_fchk=False, nocc=None):
    """
    Build the Atomic Overlap Matrices (AOMs) in the Molecular Orbitals basis. 
    """

    partition_label = format_partition(partition)
    
    try:
        S = mf.get_ovlp()
    except:
        S = mol.intor('int1e_ovlp')
    
    # Determine dimensions
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    if mo_occ is None and hasattr(mf, '_scf'):
        mo_occ = mf._scf.mo_occ
        
    is_unrest = (isinstance(mo_coeff, (list, tuple)) and len(mo_coeff) == 2) or \
                (isinstance(mo_coeff, np.ndarray) and mo_coeff.ndim == 3 and mo_coeff.shape[0] == 2)
    
    from esipy.tools import is_natorb_wf
    # Natural orbitals come from the spin-summed density
    if is_unrest and not is_fchk and is_natorb_wf(mf):
        is_unrest = False

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
            if hasattr(mf, 'ncas') and getattr(mf, 'ncas') is not None:
                n_occ_act = getattr(mf, 'ncore', 0) + getattr(mf, 'ncas', 0)
                mask_a = np.zeros(len(oa), dtype=bool); mask_a[:n_occ_act] = True
                mask_b = np.zeros(len(ob), dtype=bool); mask_b[:n_occ_act] = True
            else:
                mask_a = np.ones(len(oa), dtype=bool)
                mask_b = np.ones(len(ob), dtype=bool)
        else:
            ca, cb = mo_coeff[0], mo_coeff[1]
            oa, ob = np.asarray(mo_occ[0]), np.asarray(mo_occ[1])
            # SCF orbitals masking
            mask_a = oa > 0.5
            mask_b = ob > 0.5
            
        if partition_label in ("lowdin", "meta-lowdin", "nao", "mulliken"):
            aom_alpha, aom_beta = [], []
            if partition_label == "lowdin": U_inv = lowdin(S)
            elif partition_label == "meta-lowdin":
                from pyscf.lo import orth
                from esipy.tools import RefRHF
                ref_mfa = RefRHF(mol, ca, oa) if is_natorb else (myhf if myhf is not None else mf)
                ref_mfb = RefRHF(mol, cb, ob) if is_natorb else (myhf if myhf is not None else mf)
                # Wait, orth_ao doesn't support separate alpha/beta mf. For unrestricted, orth.orth_ao might need a UHF object.
                # Actually, in unrest, usually U_inv is built for alpha and beta separately? No, orth_ao takes mf.
                _mf = myhf if myhf is not None else (mf._scf if hasattr(mf, "_scf") else mf)
                U_inv = orth.orth_ao(_mf, method="meta-lowdin")
            elif partition_label == "nao": U_inv = nao.nao(mol, mf, S)
            
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
            # IAO logic (dev-iao): all IAO variants through build_iao_aoms
            if is_natorb:
                # Correlated wavefunctions: IAOs built from the n_act most occupied NOs of each spin
                na_e, nb_e = (mol.nelec // 2, mol.nelec - mol.nelec // 2) if isinstance(mol.nelec, int) else mol.nelec
                n_act_a = nocc if nocc is not None else na_e
                n_act_b = nocc if nocc is not None else nb_e
                coeffs_a, coeffs_b = ca[:, :n_act_a], cb[:, :n_act_b]
                occ_a_act, occ_b_act = oa[:n_act_a], ob[:n_act_b]
                print(f" | IAO for correlated wavefunctions: representing the top {n_act_a}/{n_act_b} alpha/beta NOs.")
                aom_alpha = build_iao_aoms(mol, coeffs_a, partition_label, mf=mf, S=S, c_full=coeffs_a)
                aom_beta = build_iao_aoms(mol, coeffs_b, partition_label, mf=mf, S=S, c_full=coeffs_b)
            else:
                coeffs_a = ca[:, mask_a]
                coeffs_b = cb[:, mask_b]
                aom_alpha = build_iao_aoms(mol, coeffs_a, partition_label, mf=mf, S=S)
                aom_beta = build_iao_aoms(mol, coeffs_b, partition_label, mf=mf, S=S)
            
            if is_natorb:
                aom = [[aom_alpha, aom_beta], [occ_a_act, occ_b_act]]
            else:
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
            if nocc is not None:
                mask[nocc:] = False
        else:
            occ = np.asarray(mo_occ)
            if np.any(np.isclose(occ, 1.0)):
                raise NotImplementedError(" | ROHF wavefunctions are not supported. Use UHF instead")
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
            # IAO logic (dev-iao): all IAO variants through build_iao_aoms
            if is_natorb:
                # Correlated wavefunctions: IAOs built from the n_act most occupied NOs (nocc, or half the electrons)
                n_act = nocc if nocc is not None else (mol.nelec // 2 if isinstance(mol.nelec, int) else sum(mol.nelec) // 2)
                coeffs_input = coeff_src[:, :n_act]
                occ_act = occ[:n_act]
                mol.no_coeff = coeffs_input
                print(f" | IAO for correlated wavefunctions: representing the top {n_act} NOs.")
            else:
                coeffs_input = coeff

            # For natural orbitals, c_full must match coeffs_input so the AOM dimensions match occ_act
            aom_list = build_iao_aoms(mol, coeffs_input, partition_label, mf=mf, S=S,
                                      c_full=coeffs_input if is_natorb else None)
            if is_natorb:
                aom = [aom_list, occ_act]
            else:
                aom = aom_list
                
        if save: save_file(aom, save)
        return aom
