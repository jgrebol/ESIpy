from pyscf.lo.orth import lowdin, restore_ao_character
from pyscf.lo import nao, iao
import numpy as np
from esipy.tools import save_file, format_partition, get_natorbs_from_mo, get_natorbs_from_ao, build_eta


def make_aoms(mol, mf, partition, myhf=None, save=None):
    """Build the Atomic Overlap Matrices (AOMs) in the Molecular Orbitals basis.

    Arguments:
        mol (SCF instance):
            PySCF's Mole class and helper functions to handle parameters and attributes for GTO integrals.

        mf (SCF instance):
            PySCF's object holds all parameters to control SCF.

        partition (str):
            Specifies the atom-in-molecule partition scheme. Options include 'mulliken', 'lowdin', 'meta_lowdin', 'nao', and 'iao'.

       save (str, optional, default: None):
          Sets the name of the binary file **without extension** to be stored in disk. Extension '.aoms' will be used.

    Returns:
       Smo: list
          Contains the atomic overlap matrices.
            - For restricted-SD calculations: a list of matrices with the AOMS.
            - For unrestricted-SD calculations: a list containing both alpha and beta lists of matrices as [Smo_alpha, Smo_beta].
    """

    partition = format_partition(partition)
    # UNRESTRICTED
    if (
            mf.__class__.__name__ == "UHF" or mf.__class__.__name__ == "UKS" or mf.__class__.__name__ == "SymAdaptedUHF" or mf.__class__.__name__ == "SymAdaptedUKS"):
        # Getting specific information
        S = mf.get_ovlp()
        nocc_alpha = mf.mo_occ[0].astype(int)
        nocc_beta = mf.mo_occ[1].astype(int)
        coeff_alpha = mf.mo_coeff[0][:, : nocc_alpha.sum()]
        coeff_beta = mf.mo_coeff[1][:, : nocc_beta.sum()]

        # Building the Atomic Overlap Matrices

        Smo_alpha = []
        Smo_beta = []

        if partition == "lowdin" or partition == "meta_lowdin" or partition == "nao":
            if partition == "lowdin":
                U_inv = lowdin(S)
            elif partition == "meta_lowdin":
                pre_orth_ao = restore_ao_character(mol, "ANO")
                w = np.ones(pre_orth_ao.shape[0])
                U_inv = nao._nao_sub(mol, w, pre_orth_ao, S)
            elif partition == "nao":
                U_inv = nao.nao(mol, mf, S)
            U = np.linalg.inv(U_inv)

            eta = build_eta(mol)

            for i in range(mol.natm):
                SCR_alpha = np.linalg.multi_dot((coeff_alpha.T, U.T, eta[i]))
                SCR_beta = np.linalg.multi_dot((coeff_beta.T, U.T, eta[i]))
                Smo_alpha.append(np.dot(SCR_alpha, SCR_alpha.T))
                Smo_beta.append(np.dot(SCR_beta, SCR_beta.T))

        # Special case IAO
        elif partition == "iao":
            U_alpha_iao_nonortho = iao.iao(mol, coeff_alpha)
            U_beta_iao_nonortho = iao.iao(mol, coeff_beta)
            U_alpha_inv = np.dot(U_alpha_iao_nonortho, lowdin(
                np.linalg.multi_dot((U_alpha_iao_nonortho.T, S, U_alpha_iao_nonortho))))
            U_beta_inv = np.dot(U_beta_iao_nonortho, lowdin(
                np.linalg.multi_dot((U_beta_iao_nonortho.T, S, U_beta_iao_nonortho))))
            U_alpha = np.dot(S, U_alpha_inv)
            U_beta = np.dot(S, U_beta_inv)
            pmol = iao.reference_mol(mol)

            eta = build_eta(pmol)

            for i in range(pmol.natm):
                SCR_alpha = np.linalg.multi_dot((coeff_alpha.T, U_alpha, eta[i]))
                SCR_beta = np.linalg.multi_dot((coeff_beta.T, U_beta, eta[i]))
                Smo_alpha.append(np.dot(SCR_alpha, SCR_alpha.T))
                Smo_beta.append(np.dot(SCR_beta, SCR_beta.T))

        # Special case plain Mulliken
        elif partition == "mulliken":
            eta = [np.zeros((mol.nao, mol.nao)) for i in range(mol.natm)]
            for i in range(mol.natm):
                start = mol.aoslice_by_atom()[i, -2]
                end = mol.aoslice_by_atom()[i, -1]
                eta[i][start:end, start:end] = np.eye(end - start)

            for i in range(mol.natm):
                SCR_alpha = np.linalg.multi_dot((coeff_alpha.T, S, eta[i], coeff_alpha))
                SCR_beta = np.linalg.multi_dot((coeff_beta.T, S, eta[i], coeff_beta))
                Smo_alpha.append(SCR_alpha)
                Smo_beta.append(SCR_beta)

        else:
            raise NameError("Hilbert-space scheme not available")

        Smo = [Smo_alpha, Smo_beta]

        if save:
            save_file(Smo, save)

        return Smo

    # RESTRICTED

    elif (
            mf.__class__.__name__ == "RHF" or mf.__class__.__name__ == "RKS" or mf.__class__.__name__ == "SymAdaptedRHF" or mf.__class__.__name__ == "SymAdaptedRKS"):
        # Getting specific information
        S = mf.get_ovlp()
        coeff = mf.mo_coeff[:, mf.mo_occ > 0]

        # Building the Atomic Overlap Matrices

        Smo = []

        if partition == "lowdin" or partition == "meta_lowdin" or partition == "nao":
            if partition == "lowdin":
                U_inv = lowdin(S)
            elif partition == "meta_lowdin":
                pre_orth_ao = restore_ao_character(mol, "ANO")
                w = np.ones(pre_orth_ao.shape[0])
                U_inv = nao._nao_sub(mol, w, pre_orth_ao, S)
            elif partition == "nao":
                U_inv = nao.nao(mol, mf, S)
            U = np.linalg.inv(U_inv)

            eta = build_eta(mol)

            for i in range(mol.natm):
                SCR = np.linalg.multi_dot((coeff.T, U.T, eta[i]))
                Smo.append(np.dot(SCR, SCR.T))

        # Special case IAO
        elif partition == "iao":
            U_iao_nonortho = iao.iao(mol, coeff)
            U_inv = np.dot(U_iao_nonortho, lowdin(
                np.linalg.multi_dot((U_iao_nonortho.T, S, U_iao_nonortho))))
            U = np.dot(S, U_inv)
            pmol = iao.reference_mol(mol)

            eta = build_eta(pmol)

            for i in range(pmol.natm):
                SCR = np.linalg.multi_dot((coeff.T, U, eta[i]))
                Smo.append(np.dot(SCR, SCR.T))

        # Special case plain Mulliken
        elif partition == "mulliken":
            eta = build_eta(mol)

            for i in range(mol.natm):
                SCR = np.linalg.multi_dot((coeff.T, S, eta[i], coeff))
                Smo.append(SCR)

        else:
            raise NameError("Hilbert-space scheme not available")

        if save:
            save_file(Smo, save)

        return Smo

    else:

        S = mol.intor("int1e_ovlp")
        if "CAS" in mf.__class__.__name__ or "CI" in mf.__class__.__name__:
            occ, coeff = get_natorbs_from_ao(mf, S)
        else:
            occ, coeff = get_natorbs_from_mo(mf)

        Smo = []
        if partition == "lowdin" or partition == "meta_lowdin" or partition == "nao":
            if partition == "lowdin":
                U_inv = lowdin(S)
            elif partition == "meta_lowdin":
                # Borrowed from PySCF's meta-Lowdin routine
                pre_orth_ao = restore_ao_character(mol, "ANO")
                w = np.ones(pre_orth_ao.shape[0])
                U_inv = nao._nao_sub(mol, w, pre_orth_ao, S)
            elif partition == "nao":
                U_inv = nao.nao(mol, mf, S)
            U = np.linalg.inv(U_inv)

            eta = build_eta(mol)

            for i in range(mol.natm):
                SCR = np.linalg.multi_dot((coeff.T, U.T, eta[i]))
                Smo.append(np.dot(SCR, SCR.T))

            # Special case IAO
        elif partition == "iao":
            # HF instance required to build the orthogonalization matrix
            if myhf is None:
                raise NameError(
                    " | Could not calculate partition from Natural Orbitals calculation \n | Please provide HF reference object in 'myhf'")
            coeff_hf = myhf.mo_coeff
            U_iao_nonortho = iao.iao(mol, coeff_hf)
            U_inv = np.dot(U_iao_nonortho, lowdin(
                np.linalg.multi_dot((U_iao_nonortho.T, S, U_iao_nonortho))))
            U = np.dot(S, U_inv)
            pmol = iao.reference_mol(mol)

            eta = build_eta(mol)

            for i in range(pmol.natm):
                SCR = np.linalg.multi_dot((coeff.T, U.T, eta[i]))
                Smo.append(np.dot(SCR, SCR.T))

            # Special case plain Mulliken
        elif partition == "mulliken":

            eta = build_eta(mol)

            for i in range(mol.natm):
                SCR = np.linalg.multi_dot((coeff.T, S, eta[i], coeff))
                Smo.append(SCR)

        else:
            raise NameError("Hilbert-space scheme not available")

        if save:
            save_file(Smo, save)

        return [Smo, occ]




