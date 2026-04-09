import numpy as np
from pyscf.lo import nao, orth
from pyscf import scf

from esipy.tools import save_file, format_partition, get_natorbs, build_eta

def make_aoms(mol, mf, partition, myhf=None, save=None):
    """
    Build the Atomic Overlap Matrices (AOMs) in the Molecular Orbitals basis. 
    """

    partition = format_partition(partition)
    
    try:
        S = mf.get_ovlp()
    except:
        S = mol.intor('int1e_ovlp')

    def get_iao_aoms(c, pmol_in=None):
        from pyscf.lo import iao
        U_iao_nonortho = iao.iao(mol, c)
        S_iao = np.linalg.multi_dot((U_iao_nonortho.T, S, U_iao_nonortho))
        U_inv = np.dot(U_iao_nonortho, orth.lowdin(S_iao))
        U = np.dot(S, U_inv)
        
        from pyscf.lo.iao import reference_mol
        pmol = pmol_in if pmol_in is not None else reference_mol(mol)
        eta = build_eta(pmol)
        return [np.linalg.multi_dot((c.T, U, eta[i], U.T, c)) for i in range(pmol.natm)]

    # 1. UNRESTRICTED
    if isinstance(mf, scf.uhf.UHF) or (hasattr(mf, "__name__") and mf.__name__ == "UHF"):
        ca, cb = mf.mo_coeff
        oa, ob = mf.mo_occ
        coeff_alpha = ca[:, oa > 0]
        coeff_beta = cb[:, ob > 0]

        if partition in ("lowdin", "meta_lowdin", "nao", "mulliken"):
            aom_alpha, aom_beta = [], []
            if partition == "lowdin":
                U_inv = orth.lowdin(S)
            elif partition == "meta_lowdin":
                U_inv = orth.orth_ao(mol, method='meta_lowdin')
            elif partition == "nao":
                U_inv = nao.nao(mol, mf, S)
            
            if partition == "mulliken":
                eta = build_eta(mol)
                for i in range(mol.natm):
                    aom_alpha.append(np.linalg.multi_dot((coeff_alpha.T, S, eta[i], coeff_alpha)))
                    aom_beta.append(np.linalg.multi_dot((coeff_beta.T, S, eta[i], coeff_beta)))
            else:
                U = np.dot(S, U_inv)
                eta = build_eta(mol)
                for i in range(mol.natm):
                    aom_alpha.append(np.linalg.multi_dot((coeff_alpha.T, U, eta[i], U.T, coeff_alpha)))
                    aom_beta.append(np.linalg.multi_dot((coeff_beta.T, U, eta[i], U.T, coeff_beta)))
        elif partition == "iao":
            aom_alpha = get_iao_aoms(coeff_alpha)
            aom_beta = get_iao_aoms(coeff_beta)
        else:
            raise NameError(f"Hilbert-space scheme {partition} not available")
        
        aom = [aom_alpha, aom_beta]
        if save: save_file(aom, save)
        return aom

    # 2. RESTRICTED / NATURAL ORBITALS
    else:
        # Check if it's a correlated wavefunction / needs Natural Orbitals
        is_no = False
        # If it's CCSD or CASSCF or similar, it won't be instance of scf.hf.RHF but might have Natural Orbitals
        if not isinstance(mf, scf.hf.RHF):
            is_no = True
        elif hasattr(mf, 'mo_occ') and mf.mo_occ is None:
            is_no = True
        
        if is_no:
            occ, coeff = get_natorbs(mf, S)
            # Filter non-zero occupations
            mask = np.diag(occ) > 1e-10
            coeff = coeff[:, mask]
            occ_filtered = occ[np.ix_(mask, mask)]
        else:
            coeff = mf.mo_coeff[:, mf.mo_occ > 0]

        if partition in ("lowdin", "meta_lowdin", "nao", "mulliken"):
            aom = []
            if partition == "lowdin":
                U_inv = orth.lowdin(S)
            elif partition == "meta_lowdin":
                U_inv = orth.orth_ao(mol, method='meta_lowdin')
            elif partition == "nao":
                if myhf is not None:
                    U_inv = nao.nao(mol, myhf, S)
                else:
                    U_inv = nao.nao(mol, mf, S)
            
            if partition == "mulliken":
                eta = build_eta(mol)
                for i in range(mol.natm):
                    aom.append(np.linalg.multi_dot((coeff.T, S, eta[i], coeff)))
            else:
                U = np.dot(S, U_inv)
                eta = build_eta(mol)
                for i in range(mol.natm):
                    aom.append(np.linalg.multi_dot((coeff.T, U, eta[i], U.T, coeff)))
        elif partition == "iao":
            aom = get_iao_aoms(coeff)
        else:
            raise NameError(f"Hilbert-space scheme {partition} not available")
            
        if is_no:
            res = [aom, occ_filtered]
        else:
            res = aom

        if save: save_file(res, save)
        return res
