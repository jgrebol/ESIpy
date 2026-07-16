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
    
    def get_iao_aoms(p_type, coeffs, current_mf, iaoref='minao', c_full=None):
        from pyscf.lo import iao as pyscf_iao
        from pyscf.lo import orth
        from pyscf import gto
        
        pmol = gto.M(atom=mol.atom, basis=iaoref, cart=mol.cart)
        
        # Build IAO from coeffs (which might be truncated to n_ref for natural orbitals)
        C_iao = pyscf_iao.iao(mol, coeffs, minao=iaoref)
        U = orth.vec_lowdin(C_iao, S)
        U_ao = S @ U
        proj_c = c_full if c_full is not None else coeffs
        # Get atom slices from pmol (reference basis)
        return [proj_c.T @ U_ao[:, start:end] @ U_ao[:, start:end].T @ proj_c for start, end in pmol.aoslice_by_atom()[:, -2:]]

    # Determine dimensions
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
                U_inv = orth.orth_ao(myhf if myhf is not None else mf, method="meta-lowdin")
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
            # IAO logic
            from pyscf import gto
            pmol = gto.M(atom=mol.atom, basis=iaoref, cart=mol.cart)
            n_ref = pmol.nao
            
            # Truncate natural orbitals to minimal basis size if is_natorb
            if is_natorb:
                if hasattr(mf, 'ncas') and getattr(mf, 'ncas') is not None:
                    n_occ_act = getattr(mf, 'ncore', 0) + getattr(mf, 'ncas', 0)
                    coeffs_a = ca[:, :n_occ_act]
                    coeffs_b = cb[:, :n_occ_act]
                    occ_a_act = oa[:n_occ_act]
                    occ_b_act = ob[:n_occ_act]
                else:
                    coeffs_a = ca[:, :n_ref]
                    coeffs_b = cb[:, :n_ref]
                    occ_a_act = oa[:n_ref]
                    occ_b_act = ob[:n_ref]
            else:
                coeffs_a = ca[:, mask_a]
                coeffs_b = cb[:, mask_b]

            aom_alpha = get_iao_aoms(partition_label, coeffs_a, mf, iaoref=iaoref, c_full=ca[:, mask_a])
            aom_beta = get_iao_aoms(partition_label, coeffs_b, mf, iaoref=iaoref, c_full=cb[:, mask_b])
            
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
                _mf = myhf if myhf is not None else (mf._scf if hasattr(mf, "_scf") else mf)
                U_inv = orth.orth_ao(_mf, method="meta-lowdin")
            elif partition_label == "nao": U_inv = nao.nao(mol, mf, S)
            
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
            from pyscf import gto
            pmol = gto.M(atom=mol.atom, basis=iaoref, cart=mol.cart)
            n_ref = pmol.nao
            
            if is_natorb:
                if hasattr(mf, 'ncas') and getattr(mf, 'ncas') is not None:
                    n_occ_act = getattr(mf, 'ncore', 0) + getattr(mf, 'ncas', 0)
                    coeffs_input = coeff_src[:, :n_occ_act]
                    occ_act = occ[:n_occ_act]
                else:
                    coeffs_input = coeff_src[:, :n_ref]
                    occ_act = occ[:n_ref]
            else:
                coeffs_input = coeff

            # For nat orbs, c_full must match coeffs_input so AOM dimensions match occ_act.
            # Passing the full coeff (all masked orbs) would create a shape mismatch with occ_act.
            aom_list = get_iao_aoms(partition_label, coeffs_input, mf, iaoref=iaoref,
                                    c_full=coeffs_input if is_natorb else coeff)
            if is_natorb:
                aom = [aom_list, occ_act]
            else:
                aom = aom_list
                
        if save: save_file(aom, save)
        return aom
