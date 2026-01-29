import numpy as np
from pyscf.lo import nao
from pyscf.lo.orth import lowdin
from pyscf import scf

from esipy.tools import save_file, format_partition, get_natorbs, build_eta

def make_aoms(mol, mf, partition, myhf=None, save=None):
    """
    Build the Atomic Overlap Matrices (AOMs) in the Molecular Orbitals basis. If using Natural Orbitals,
    the HF instance is required as the reference to build the IAO transformation matrix.

    :param mol: PySCF Mole object.
    :type mol: pyscf.gto.Mole
    :param mf: PySCF SCF object.
    :type mf: pyscf.scf.hf.SCF
    :param partition: String with the name of the partition.
    :type partition: str
    :param myhf: PySCF SCF object. Required if using Natural Orbitals.
    :type myhf: pyscf.scf.hf.SCF, optional
    :param save: String with the name of the file to save the AOMs.
    :type save: str, optional

    :returns: For RESTRICTED calculations, a list with each of the AOMs. For UNRESTRICTED calculations, a list with the alpha and beta AOMs, as [aom_alpha, aom_beta]. For NATURAL ORBITALS calculations, a list with the AOMs and the Natural Orbitals occupation numbers, as [aom, occ].
    :rtype: list
    """

    partition = format_partition(partition)
    free_atom = False # For IAO-AUTOSAD
    # UNRESTRICTED
    if isinstance(mf, scf.uhf.UHF):
    #if mf.__class__.__name__ in ("UHF", "UKS", "SymAdaptedUHF", "SymAdaptedUKS") or (hasattr(mf, "__name__") and mf.__name__ == "UHF"):
        # Getting specific information
        S = mf.get_ovlp()
        nocc_alpha = mf.mo_occ[0].astype(int)
        nocc_beta = mf.mo_occ[1].astype(int)
        coeff_alpha = mf.mo_coeff[0][:, : nocc_alpha.sum()]
        coeff_beta = mf.mo_coeff[1][:, : nocc_beta.sum()]

        # Building the Atomic Overlap Matrices

        aom_alpha, aom_beta = [], []
        if partition in ("lowdin", "meta_lowdin", "nao"):
            if partition == "lowdin":
                U_inv = lowdin(S)
            elif partition == "meta_lowdin":
                from pyscf.lo.orth import restore_ao_character
                if hasattr(mol, '_read_fchk'):
                    prev_reorder = getattr(mol, '_reorder', True)
                    mol._reorder = False
                    # Force recompute of overlap if present
                    if hasattr(mol, '_ovlp'):
                        mol._ovlp = None
                    pre_orth_ao = restore_ao_character(mol, "ANO")
                    mol._reorder = prev_reorder
                    mol._ovlp = None
                else:
                    pre_orth_ao = restore_ao_character(mol, "ANO")
                w = np.ones(pre_orth_ao.shape[0])
                U_inv = nao._nao_sub(mol, w, pre_orth_ao, S)
            elif partition == "nao":
                U_inv = nao.nao(mol, mf, S)
            U = np.linalg.inv(U_inv)

            eta = build_eta(mol)
            for i in range(mol.natm):
                SCR_a = coeff_alpha.T @ U.T @ eta[i]
                SCR_b = coeff_beta.T @ U.T @ eta[i]
                aom_alpha.append(SCR_a @ SCR_a.T)
                aom_beta.append(SCR_b @ SCR_b.T)

        # Special case IAO
        elif partition == "iao" or partition == "iao-autosad":
            from pyscf.lo.iao import reference_mol
            if partition.startswith("iao-autosad"):
                from esipy.tools import autosad
                if partition == "iao-autosad-freeatom":
                    U_alpha_iao_nonortho, U_beta_iao_nonortho = autosad(mol, mf, free_atom=True)
                else:
                    U_alpha_iao_nonortho, U_beta_iao_nonortho = autosad(mol, mf, free_atom=False)
            else:
                from pyscf.lo.iao import iao
                U_alpha_iao_nonortho = iao(mol, coeff_alpha)
                U_beta_iao_nonortho = iao(mol, coeff_beta)
            pmol = reference_mol(mol)
            U_alpha_inv = np.dot(U_alpha_iao_nonortho, lowdin(
                np.linalg.multi_dot((U_alpha_iao_nonortho.T, S, U_alpha_iao_nonortho))))
            U_beta_inv = np.dot(U_beta_iao_nonortho, lowdin(
                np.linalg.multi_dot((U_beta_iao_nonortho.T, S, U_beta_iao_nonortho))))
            U_alpha = np.dot(S, U_alpha_inv)
            U_beta = np.dot(S, U_beta_inv)

            eta = build_eta(pmol)

            for i in range(pmol.natm):
                SCR_alpha = np.linalg.multi_dot((coeff_alpha.T, U_alpha, eta[i]))
                SCR_beta = np.linalg.multi_dot((coeff_beta.T, U_beta, eta[i]))
                aom_alpha.append(np.dot(SCR_alpha, SCR_alpha.T))
                aom_beta.append(np.dot(SCR_beta, SCR_beta.T))

        # Special case plain Mulliken
        elif partition == "mulliken":
            eta = build_eta(mol)

            for i in range(mol.natm):
                SCR_alpha = np.linalg.multi_dot((coeff_alpha.T, S, eta[i], coeff_alpha))
                SCR_beta = np.linalg.multi_dot((coeff_beta.T, S, eta[i], coeff_beta))
                aom_alpha.append(SCR_alpha)
                aom_beta.append(SCR_beta)

        else:
            raise NameError("Hilbert-space scheme not available")

        aom = [aom_alpha, aom_beta]

        if save:
            save_file(aom, save)

        return aom

    # RESTRICTED

    elif isinstance(mf, scf.hf.RHF):
#    elif mf.__class__.__name__ in ("RHF", "RKS", "SymAdaptedRHF", "SymAdaptedRKS") or (hasattr(mf, "__name__") and mf.__name__ == "RHF"):
        # Getting specific information
        S = mf.get_ovlp()
        coeff = mf.mo_coeff[:, mf.mo_occ > 0]

        # Building the Atomic Overlap Matrices

        aom = []
        if partition in ("lowdin", "meta_lowdin", "nao"):
            if partition == "lowdin":
                # Free of unitary transformations :)
                U_inv = lowdin(S)
            elif partition == "meta_lowdin":
                from pyscf.lo.orth import restore_ao_character
                pre_orth_ao = restore_ao_character(mol, "ANO")
                w = np.ones(pre_orth_ao.shape[1])
                U_inv = nao._nao_sub(mol, w, pre_orth_ao, S)
            elif partition == "nao":
                U_inv = nao.nao(mol, mf, S)

            U = np.linalg.inv(U_inv)

            eta = build_eta(mol)

            for i in range(mol.natm):
                SCR = coeff.T @ U.T @ eta[i]
                aom.append(SCR @ SCR.T)

        # Special case IAO
        elif partition.startswith("iao"):
            from pyscf.lo.iao import reference_mol
            if partition.startswith("iao-autosad"):
                from esipy.tools import autosad
                if partition == "iao-autosad-freeatom":
                    U_iao_nonortho = autosad(mol, mf, free_atom=True)
                else:
                    U_iao_nonortho = autosad(mol, mf, free_atom=False)
            else:
                from pyscf.lo.iao import iao
                U_iao_nonortho = iao(mol, coeff)
            pmol = reference_mol(mol)
            U_inv = np.dot(U_iao_nonortho, lowdin(
                np.linalg.multi_dot((U_iao_nonortho.T, S, U_iao_nonortho))))
            U = np.dot(S, U_inv)

            eta = build_eta(pmol)

            for i in range(pmol.natm):
                SCR = np.linalg.multi_dot((coeff.T, U, eta[i]))
                aom.append(np.dot(SCR, SCR.T))

        #elif partition == "iao-autosad":
        #    U_iao_nonortho = autosad(mol, mf)
        #    U_inv = np.dot(U_iao_nonortho, lowdin(
        #        np.linalg.multi_dot((U_iao_nonortho.T, S, U_iao_nonortho))))
        #    U = np.dot(S, U_inv)
        #    pmol = iao.reference_mol(mol)
#
#            eta = build_eta(pmol)
#
#            for i in range(pmol.natm):
#                SCR = np.linalg.multi_dot((coeff.T, U, eta[i]))
#                aom.append(np.dot(SCR, SCR.T))

        # Special case plain Mulliken
        elif partition == "mulliken":
            eta = build_eta(mol)

            for i in range(mol.natm):
                SCR = np.linalg.multi_dot((coeff.T, S, eta[i], coeff))
                aom.append(SCR)

        else:
            raise NameError("Hilbert-space scheme not available")

        if save:
            save_file(aom, save)

        return aom

    else:

        S = mol.intor("int1e_ovlp")
        if "fci" in mf.__module__:
            raise NameError(" | FCI not supported yet")
            #occ, coeff = get_natorbs_fci(mf, S, myhf, 2, 2)
        else:
            occ, coeff = get_natorbs(mf, S)

        aom = []
        if partition in ("lowdin", "meta_lowdin", "nao"):
            if partition == "lowdin":
                U_inv = lowdin(S)
            elif partition == "meta_lowdin":
                from pyscf.lo.orth import restore_ao_character
                pre_orth_ao = restore_ao_character(mol, "ANO")
                w = np.ones(pre_orth_ao.shape[0])
                U_inv = nao._nao_sub(mol, w, pre_orth_ao, S)
            elif partition == "nao":
                # NAOs must be built from HF reference
                coeff_hf = myhf.mo_coeff
                if len(np.shape(coeff_hf)) == 3:
                    coeff_hf = np.sum(coeff_hf, axis=0)
                if myhf is None:
                    raise NameError(
                        " | Could not calculate partition from Natural Orbitals calculation \n | Please provide HF reference object in 'myhf'")
                U_inv = nao.nao(mol, mf, S)
            U = np.linalg.inv(U_inv)

            eta = build_eta(mol)

            for i in range(mol.natm):
                SCR = coeff.T @ U.T @ eta[i]
                aom.append(SCR @ SCR.T)

            # Special case IAO
        elif partition == "iao":
            # HF instance required to build the orthogonalization matrix
            if myhf is None:
                raise NameError(
                    " | Could not calculate partition from Natural Orbitals calculation \n | Please provide HF reference object in 'myhf'")
            from pyscf.lo.iao import iao
            coeff_hf = myhf.mo_coeff
            if len(np.shape(coeff_hf)) == 3:
                coeff_hf = np.sum(coeff_hf, axis=0)
            U_iao_nonortho = iao(mol, coeff_hf)
            U_inv = np.dot(U_iao_nonortho, lowdin(
                np.linalg.multi_dot((U_iao_nonortho.T, S, U_iao_nonortho))))
            U = np.dot(S, U_inv)
            from pyscf.lo.iao import reference_mol
            pmol = reference_mol(mol)

            eta = build_eta(mol)

            for i in range(pmol.natm):
                SCR = np.linalg.multi_dot((coeff.T, U.T, eta[i]))
                aom.append(np.dot(SCR, SCR.T))

            # Special case plain Mulliken
        elif partition == "mulliken":

            eta = build_eta(mol)

            for i in range(mol.natm):
                SCR = np.linalg.multi_dot((coeff.T, S, eta[i], coeff))
                aom.append(SCR)

        else:
            raise NameError("Hilbert-space scheme not available")

        aom = [aom, occ]

        if save:
            save_file(aom, save)

        return aom
