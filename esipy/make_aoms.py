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
        try:
            from esipy import iao as esipy_iao
        except ImportError:
            try:
                import iao as esipy_iao
            except ImportError:
                # Fallback to simple pyscf iao if iao.py is missing
                from pyscf.lo import iao as pyscf_iao
                from pyscf.lo import orth
                C_iao = pyscf_iao.iao(mol, coeffs)
                U = orth.vec_lowdin(C_iao, S)
                U_ao = S @ U
                proj_c = c_full if c_full is not None else coeffs
                return [proj_c.T @ U_ao[:, start:end] @ U_ao[:, start:end].T @ proj_c for start, end in mol.aoslice_by_atom()[:, -2:]]

        # Use the new methodology from iao.py
        p_type = p_type.lower()
        if p_type == 'peiao':
            C_iao, pmol = esipy_iao.peiao(mol, coeffs, source_basis=iaoref)
        elif p_type == 'effao':
            C_iao, pmol = esipy_iao.effao(mol, coeffs, source_basis=iaoref)
        elif p_type == 'fpiao':
            C_iao, pmol = esipy_iao.fpiao(mol, coeffs, source_basis=iaoref)
        else:
            C_iao, pmol = esipy_iao.iao(mol, coeffs, source_basis=iaoref)
        
        # C_iao is in AO basis. Orthonormal.
        U_ao = S @ C_iao
        proj_c = c_full if c_full is not None else coeffs
        # Get atom slices from pmol (reference basis)
        return [proj_c.T @ U_ao[:, start:end] @ U_ao[:, start:end].T @ proj_c for start, end in pmol.aoslice_by_atom()[:, -2:]]

    # Determine dimensions
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    is_unrest = (isinstance(mo_coeff, (list, tuple)) and len(mo_coeff) == 2) or \
                (isinstance(mo_coeff, np.ndarray) and mo_coeff.ndim == 3 and mo_coeff.shape[0] == 2)
    
    # 1. UNRESTRICTED
    if is_unrest:
        ca, cb = mo_coeff[0], mo_coeff[1]
        oa, ob = np.asarray(mo_occ[0]), np.asarray(mo_occ[1])
        is_natorb = False
        
        from esipy.tools import is_natorb_wf
        if is_fchk:
            d_lbl = getattr(mf, 'density_label', '')
            if 'MP2' in d_lbl or 'CC' in d_lbl or 'CI' in d_lbl:
                is_natorb = True
            elif 'SCF' in d_lbl:
                is_natorb = False
        else:
            if is_natorb_wf(mf):
                is_natorb = True

        if partition_label in ("lowdin", "meta-lowdin", "nao", "mulliken"):
            aom_alpha, aom_beta = [], []
            if partition_label == "lowdin": U_inv = lowdin(S)
            elif partition_label == "meta-lowdin":
                from pyscf.lo import orth
                U_inv = orth.orth_ao(mol, method="meta-lowdin")
            elif partition_label == "nao": U_inv = nao.nao(mol, mf, S)
            
            mask_a = oa > 1e-10 if is_natorb else oa > 0.5
            mask_b = ob > 1e-10 if is_natorb else ob > 0.5
            
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
            mask_a = oa > 1e-10 if is_natorb else oa > 0.5
            mask_b = ob > 1e-10 if is_natorb else ob > 0.5
            aom_alpha = get_iao_aoms(partition_label, ca[:, mask_a], mf, iaoref=iaoref, c_full=ca[:, mask_a])
            aom_beta = get_iao_aoms(partition_label, cb[:, mask_b], mf, iaoref=iaoref, c_full=cb[:, mask_b])
            if is_natorb:
                aom = [[aom_alpha, aom_beta], [oa[mask_a], ob[mask_b]]]
            else:
                aom = [aom_alpha, aom_beta]
        
        if save: save_file(aom, save)
        return aom

    # 2. RESTRICTED
    else:
        mo_coeff = mf.mo_coeff
        mo_occ = np.asarray(mf.mo_occ)
        if isinstance(mo_coeff, (list, tuple)) and len(mo_coeff) == 1:
            mo_coeff = mo_coeff[0]
            mo_occ = mo_occ[0]
            
        occ = mo_occ
        is_natorb = False
        if occ is not None and np.asarray(occ).dtype != object:
            is_natorb = np.any((occ > 1e-6) & (np.abs(occ - 1.0) > 1e-6) & (np.abs(occ - 2.0) > 1e-6))
        
        if is_fchk:
            d_lbl = getattr(mf, 'density_label', '')
            if 'MP2' in d_lbl or 'CC' in d_lbl or 'CI' in d_lbl:
                is_natorb = True
            elif 'SCF' in d_lbl:
                is_natorb = False
        else:
            from esipy.tools import is_natorb_wf
            if is_natorb_wf(mf):
                is_natorb = True
                occ, mo_coeff = get_natorbs(mf, S)
                
        coeff_src = mo_coeff

        if partition_label in ("lowdin", "meta-lowdin", "nao", "mulliken"):
            mask = (occ > 1e-10) if is_natorb else (occ > 0.5)
            coeff = coeff_src[:, mask]
            aom = []
            if partition_label == "lowdin": U_inv = lowdin(S)
            elif partition_label == "meta-lowdin":
                from pyscf.lo import orth
                U_inv = orth.orth_ao(myhf if myhf is not None else mf, method="meta-lowdin")
            elif partition_label == "nao": U_inv = nao.nao(mol, mf, S)
            
            if partition_label == "mulliken":
                eta = build_eta(mol)
                for i in range(mol.natm): aom.append(np.linalg.multi_dot((coeff.T, S, eta[i], coeff)))
            else:
                U = np.linalg.inv(U_inv); eta = build_eta(mol)
                for i in range(mol.natm): aom.append(coeff.T @ U.T @ eta[i] @ U @ coeff)
            
            if is_natorb and (not is_fchk or np.any(np.abs(occ - 2.0) > 1e-6)):
                aom = [aom, occ[mask]]
        else:
            # IAO logic
            mask = (occ > 1e-10) if is_natorb else (occ > 0.5)
            coeff = coeff_src[:, mask]

            aom_list = get_iao_aoms(partition_label, coeff, mf, iaoref=iaoref, c_full=coeff)
            if is_natorb:
                aom = [aom_list, occ[mask]]
            else:
                aom = aom_list
                
        if save: save_file(aom, save)
        return aom
