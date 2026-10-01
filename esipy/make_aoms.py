from pyscf import lo
from pyscf.lo.orth import lowdin
import numpy as np
from esipy.tools import save_file, format_partition, get_natorbs, is_natorb_wf, RefRHF, build_iao_aoms


def make_aoms(mol, mf, partition, save=None, nocc=None, myhf=None, is_fchk=False):
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
            - For natural orbitals calculations: [Smo, occ].
    """

    partition = format_partition(partition)
    try:
        S = mf.get_ovlp()
    except Exception:
        S = mol.intor_symmetric('int1e_ovlp')

    # CORRELATED / NATURAL ORBITALS
    if is_natorb_wf(mf) or getattr(mf, 'is_natorb', False):
        occ, coeff = get_natorbs(mf, S)
        mol.no_coeff = coeff
        n_act = nocc if nocc is not None else (mol.nelec // 2 if isinstance(mol.nelec, int) else sum(mol.nelec) // 2)
        coeff_iao = coeff[:, :n_act]

        c_proj = coeff[:, :nocc] if nocc is not None else coeff
        occ_proj = occ[:nocc] if nocc is not None else occ

        Smo = []

        if partition in ("lowdin", "meta_lowdin", "meta-lowdin", "nao"):
            if partition == "lowdin":
                U_inv = lowdin(S)
            else:
                ref_mf = RefRHF(mol, coeff, occ)
                U_inv = lo.orth_ao(ref_mf, "meta-lowdin" if "meta" in partition else "nao", s=S)
            U = np.linalg.inv(U_inv)

            eta = [np.zeros((mol.nao, mol.nao)) for i in range(mol.natm)]
            for i in range(mol.natm):
                start = mol.aoslice_by_atom()[i, -2]
                end = mol.aoslice_by_atom()[i, -1]
                eta[i][start:end, start:end] = np.eye(end - start)

            for i in range(mol.natm):
                SCR = np.linalg.multi_dot((c_proj.T, U.T, eta[i]))
                Smo.append(np.dot(SCR, SCR.T))

        elif "iao" in partition or "piao" in partition:
            mol.no_coeff = coeff_iao
            Smo = build_iao_aoms(mol, coeff_iao, partition, mf=mf, S=S, c_full=coeff_iao)
            occ_proj = occ[:n_act]

        elif partition == "mulliken":
            eta = [np.zeros((mol.nao, mol.nao)) for i in range(mol.natm)]
            for i in range(mol.natm):
                start = mol.aoslice_by_atom()[i, -2]
                end = mol.aoslice_by_atom()[i, -1]
                eta[i][start:end, start:end] = np.eye(end - start)

            for i in range(mol.natm):
                SCR = np.linalg.multi_dot((c_proj.T, S, eta[i], c_proj))
                Smo.append(SCR)

        else:
            raise NameError("Hilbert-space scheme not available")

        Smo = [Smo, occ_proj]

        if save:
            save_file(Smo, save)

        return Smo

    # UNRESTRICTED
    elif (
            mf.__class__.__name__ == "UHF" or mf.__class__.__name__ == "UKS" or mf.__class__.__name__ == "SymAdaptedUHF" or mf.__class__.__name__ == "SymAdaptedUKS" or (hasattr(mf, 'mo_coeff') and isinstance(mf.mo_coeff, (list, tuple)) and len(mf.mo_coeff) == 2)):
        occ_coeff_alpha = mf.mo_coeff[0][:, np.asarray(mf.mo_occ[0]) > 0.5]
        occ_coeff_beta = mf.mo_coeff[1][:, np.asarray(mf.mo_occ[1]) > 0.5]

        Smo_alpha = []
        Smo_beta = []

        if partition in ("lowdin", "meta_lowdin", "meta-lowdin", "nao"):
            if partition == "lowdin":
                U_inv = lowdin(S)
            else:
                U_inv = lo.orth_ao(mf, "meta-lowdin" if "meta" in partition else "nao", s=S)
            U = np.linalg.inv(U_inv)

            eta = [np.zeros((mol.nao, mol.nao)) for i in range(mol.natm)]
            for i in range(mol.natm):
                start = mol.aoslice_by_atom()[i, -2]
                end = mol.aoslice_by_atom()[i, -1]
                eta[i][start:end, start:end] = np.eye(end - start)

            for i in range(mol.natm):
                SCR_alpha = np.linalg.multi_dot((occ_coeff_alpha.T, U.T, eta[i]))
                SCR_beta = np.linalg.multi_dot((occ_coeff_beta.T, U.T, eta[i]))
                Smo_alpha.append(np.dot(SCR_alpha, SCR_alpha.T))
                Smo_beta.append(np.dot(SCR_beta, SCR_beta.T))

        elif "iao" in partition or "piao" in partition:
            Smo_alpha = build_iao_aoms(mol, occ_coeff_alpha, partition, mf=mf, S=S)
            Smo_beta = build_iao_aoms(mol, occ_coeff_beta, partition, mf=mf, S=S)

        elif partition == "mulliken":
            eta = [np.zeros((mol.nao, mol.nao)) for i in range(mol.natm)]
            for i in range(mol.natm):
                start = mol.aoslice_by_atom()[i, -2]
                end = mol.aoslice_by_atom()[i, -1]
                eta[i][start:end, start:end] = np.eye(end - start)

            for i in range(mol.natm):
                SCR_alpha = np.linalg.multi_dot((occ_coeff_alpha.T, S, eta[i], occ_coeff_alpha))
                SCR_beta = np.linalg.multi_dot((occ_coeff_beta.T, S, eta[i], occ_coeff_beta))
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
            mf.__class__.__name__ == "RHF" or mf.__class__.__name__ == "RKS" or mf.__class__.__name__ == "SymAdaptedRHF" or mf.__class__.__name__ == "SymAdaptedRKS" or hasattr(mf, 'mo_coeff')):
        occ_coeff = mf.mo_coeff[:, np.asarray(mf.mo_occ) > 0.5]

        Smo = []

        if partition in ("lowdin", "meta_lowdin", "meta-lowdin", "nao"):
            if partition == "lowdin":
                U_inv = lowdin(S)
            else:
                U_inv = lo.orth_ao(mf, "meta-lowdin" if "meta" in partition else "nao", s=S)
            U = np.linalg.inv(U_inv)

            eta = [np.zeros((mol.nao, mol.nao)) for i in range(mol.natm)]
            for i in range(mol.natm):
                start = mol.aoslice_by_atom()[i, -2]
                end = mol.aoslice_by_atom()[i, -1]
                eta[i][start:end, start:end] = np.eye(end - start)

            for i in range(mol.natm):
                SCR = np.linalg.multi_dot((occ_coeff.T, U.T, eta[i]))
                Smo.append(np.dot(SCR, SCR.T))

        elif "iao" in partition or "piao" in partition:
            Smo = build_iao_aoms(mol, occ_coeff, partition, mf=mf, S=S)

        elif partition == "mulliken":
            eta = [np.zeros((mol.nao, mol.nao)) for i in range(mol.natm)]
            for i in range(mol.natm):
                start = mol.aoslice_by_atom()[i, -2]
                end = mol.aoslice_by_atom()[i, -1]
                eta[i][start:end, start:end] = np.eye(end - start)

            for i in range(mol.natm):
                SCR = np.linalg.multi_dot((occ_coeff.T, S, eta[i], occ_coeff))
                Smo.append(SCR)

        else:
            raise NameError("Hilbert-space scheme not available")

        if save:
            save_file(Smo, save)

        return Smo

    else:
        print(" Only restricted and unrestricted HF and KS-DFT available with this version of the program")
        return
