import numpy as np
import re
from pyscf.lo import nao
from pyscf.lo.orth import lowdin
from pyscf import scf

from esipy.tools import save_file, format_partition, build_eta

def make_aoms(mol, mf, partition, myhf=None, save=None):
    """
    Build the Atomic Overlap Matrices (AOMs) in the Molecular Orbitals basis. 
    """

    partition_label = format_partition(partition)
    
    try:
        S = mf.get_ovlp()
    except:
        S = mol.intor('int1e_ovlp')
    
    def get_iao_aoms(p_type, c, current_mf):
        from pyscf.lo import iao as pyscf_iao
        from pyscf.lo import orth
        
        # Standard IAO from PySCF
        C_iao_nonorth = pyscf_iao.iao(mol, c)
        U_nonorth = orth.vec_lowdin(C_iao_nonorth, S)
        
        U = np.dot(S, U_nonorth)
        eta = build_eta(mol)
        return [np.linalg.multi_dot((c.T, U, eta[i], U.T, c)) for i in range(mol.natm)]

    # 1. UNRESTRICTED
    is_unrest = isinstance(mf, scf.uhf.UHF) or (hasattr(mf, "__name__") and mf.__name__ == "UHF")
    if is_unrest and isinstance(mf.mo_coeff, (list, tuple)) and len(mf.mo_coeff) == 2:
        ca, cb = mf.mo_coeff
        oa, ob = np.asarray(mf.mo_occ[0]), np.asarray(mf.mo_occ[1])
        
        coeff_alpha = ca[:, oa > 0.5]
        coeff_beta = cb[:, ob > 0.5]
        
        if partition_label in ("lowdin", "meta_lowdin", "nao", "mulliken"):
            aom_alpha, aom_beta = [], []
            if partition_label == "lowdin": U_inv = lowdin(S)
            elif partition_label == "meta_lowdin":
                from pyscf.lo import orth
                U_inv = orth.orth_ao(mf, method="meta_lowdin")
            elif partition_label == "nao": U_inv = nao.nao(mol, mf, S)
            
            if partition_label == "mulliken":
                eta = build_eta(mol)
                for i in range(mol.natm):
                    aom_alpha.append(np.linalg.multi_dot((coeff_alpha.T, S, eta[i], coeff_alpha)))
                    aom_beta.append(np.linalg.multi_dot((coeff_beta.T, S, eta[i], coeff_beta)))
            else:
                U = np.linalg.inv(U_inv); eta = build_eta(mol)
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
    else:
        occ = np.asarray(mf.mo_occ)
        mask = occ > 0.5
        # Robust source handling
        coeff_src = mf.mo_coeff[0] if isinstance(mf.mo_coeff, (list, tuple)) else mf.mo_coeff
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
