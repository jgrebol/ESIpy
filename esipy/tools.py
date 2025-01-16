import pickle
from functools import reduce

import numpy as np
from os import environ

environ["NUMEXPR_NUM_THREADS"] = "1"
environ["OMP_NUM_THREADS"] = "1"
environ["MKL_NUM_THREADS"] = "1"


def wf_type(Smo):
    """
    Checks the topology of the AOMs to obtain the type of wavefunction.
    Args:
        Smo: The Atomic Overlap Matrices (AOMs) in the MO basis.

    Returns:
        A string with the type of wave function ('rest', 'unrest' or 'no').
    """
    if isinstance(Smo[0][0][0], float):
        return "rest"
    elif isinstance(Smo[1][0][0], float):
        return "no"
    elif isinstance(Smo[1][0][0][0], float):
        return "unrest"
    else:
        raise NameError("Could not find the type of wave function from the AOMs")


def find_multiplicity(Smo):
    """
    Checks the topology of the AOMs to get the difference between alpha and beta electrons.

    Arguments:
       Smo: The Atomic Overlap Matrices (AOMs) in the MO basis.

    Returns:
       A string with the multiplicity of the calculations.
    """

    if len(Smo[0][0]) == len(Smo[1][0]):
        return "singlet"
    elif len(Smo[0][0]) == (len(Smo[1][0]) + 1):
        return "doublet"
    elif len(Smo[0][0]) == (len(Smo[1][0]) + 2):
        return "triplet"
    elif len(Smo[0][0]) == (len(Smo[1][0]) + 3):
        return "quadruplet"
    elif len(Smo[0][0]) == (len(Smo[1][0]) + 4):
        return "quintuplet"
    else:
        return None

def load_file(file):
    """Loads the data from a file using Pickle.

    Args:
       file: string
          Contains the name or the path containing the variable.

    Returns:
       file:
          The file loaded.
    """
    from pickle import load
    with open(file, "rb") as f:
        return load(f)

def save_file(file, save):
    """Loads the variable into a file.

    Args:
         file:
             Contains the variable to be stored.
         Smo: string
            Contains the name or the path containing the AOMs.

    Returns:
       Smo: list of matrices
          The AOMs required for the ESIpy code.
    """
    from pickle import dump
    with open(save, "wb") as f:
        dump(file, f)


def find_distances(arr, geom):
    """
    Collects the distance between the atoms in ring connectivity.
    Args:
        arr: Indices of the atoms in ring connectivity.
        mol: PySCF's 'mol' object. Units must be in Angstroms, which is the default PySCF option.
        geom: Geometry of the molecule as in mol.atom_coords().

    Returns:
        List containing the distances in of the members of the ring in Bohrs.
    """
    distances = []
    for i in range(len(arr)):
        coord1 = geom[arr[i] - 1]
        coord2 = geom[arr[(i + 1) % len(arr)] - 1]
        distances.append(np.linalg.norm(coord1 - coord2) * 0.529177249)
    return distances


def find_dis(arr, Smo):
    """
    Collects the DIs between the atoms in ring connectivity.
    Args:
        arr: Indices of the atoms in ring connectivity.
        Smo: The Atomic Overlap Matrices (AOMs) in the MO basis.

    Returns:
        List containing the DIs in of the members of the ring
    """
    return [4 * np.trace(np.dot(Smo[arr[i] - 1], Smo[arr[(i + 1) % len(arr)] - 1])) for i in range(len(arr))]


def find_di(Smo, i, j):
    """
    Collects the DI between two atoms.
    Args:
        Smo: The Atomic Overlap Matrices (AOMs) in the MO basis.
        i: Index of the first atom.
        j: Index of the second atom.

    Returns:
        DI between the atoms i and j.

    """
    return 2 * np.trace(np.dot(Smo[i - 1], Smo[j - 1]))


def find_di_no(Smo, i, j):
    """
    Collects the DI between two atoms for Natural Orbitals calcualtion.
    Args:
        Smo: The Atomic Overlap Matrices (AOMs) in the MO basis.
        i: Index of the first atom.
        j: Index of the second atom.

    Returns:
        DI between the atoms i and j.
    """
    return np.trace(np.linalg.multi_dot((Smo[1], Smo[0][i - 1], Smo[1], Smo[0][j - 1])))


def find_lis(arr, Smo):
    """
    Collects the LIs between the atoms in ring connectivity.
    Args:
        arr: Indices of the atoms in ring connectivity.
        Smo: The Atomic Overlap Matrices (AOMs) in the MO basis.

    Returns:
        List containing the LIs in of the members of the ring
    """
    return [2 * np.trace(np.dot(Smo[arr[i] - 1], Smo[arr[i] - 1])) for i in range(len(arr))]


def find_ns(arr, Smo):
    """
    Collects the atomic populations of all the atoms in the ring.
    Args:
        arr: Indices of the atoms in ring connectivity.
        Smo: The Atomic Overlap Matrices (AOMs) in the MO basis.

    Returns:
        List containing the atomic populations of the members of the ring.
    """
    return [2 * np.trace(Smo[arr[i] - 1]) for i in range(len(arr))]


def av1245_pairs(arr):
    """
    Collects the series of atoms that fulfill the 1-2-4-5 relationship for the AV1245 calculation.
    Args:
        arr: Indices of the atoms in ring connectivity.

    Returns:
        List containing the series of atoms that fulfill the 1-2-4-5 relationship
    """
    return [(arr[i % len(arr)], arr[(i + 1) % len(arr)], arr[(i + 3) % len(arr)], arr[(i + 4) % len(arr)])
            for i in range(len(arr))]


def mol_info(mol=None, mf=None, save=None, partition=None):
    """Obtains information from the molecule and the calculation
    to complement the main code function without requiring the 'mol'
    and 'mf' objects.
    Args:
        mol: PySCF Mole object.
        mf: PySCF SCF object.
        save: String with the name of the file to save the information.
        partition: String with the name of the partition.
    Returns:
        Dictionary with the information of the molecule and the calculation.
    """
    info = {}
    info.update({"partition": partition})
    if mol:
        info.update({
            "symbols": [mol.atom_symbol(i) for i in range(mol.natm)],
            "atom_numbers": [i + 1 for i in range(mol.natm)],
            "basisset": mol.basis,
            "geom": mol.atom_coords()
        })

    if mf:
        info.update({
            "calctype": mf.__class__.__name__,
            "energy": mf.e_tot,
            "method": mf.__module__
        })

        if "dft" in mf.__module__ and mf.xc is not None:
            info.update({"xc": mf.xc})
        else:
            info.update({"xc": "None"})

    if save:
        save_file(info, save)

    return info


def format_partition(partition):
    """
    Filters the 'partition' attribute for flexibility
    Args:
        partition: String with the name of the partition.

    Returns:
        String with the standard partition name for ESIpy.
    """
    partition = partition.lower()
    if partition in ["m", "mul", "mulliken"]:
        return "mulliken"
    elif partition in ["l", "low", "lowdin"]:
        return "lowdin"
    elif partition in ["ml", "mlow", "m-low", "meta-low", "metalow", "mlowdin", "m-lowdin", "metalowdin", "meta-lowdin",
                       "meta_lowdin"]:
        return "meta_lowdin"
    elif partition in ["n", "nao", "natural", "nat"]:
        return "nao"
    elif partition in ["i", "iao", "intrinsic", "intr"]:
        return "iao"
    else:
        raise NameError(" | Invalid partition scheme")


def format_short_partition(partition):
    """
    Filters the short version of the 'partition' attribute.
    Args:
        partition: String with the name of the partition.

    Returns:
        String with the short version of the partition scheme.
    """
    partition = partition.lower()
    if partition == "mulliken":
        return "mul"
    elif partition == "lowdin":
        return "low"
    elif partition == "meta_lowdin":
        return "metalow"
    elif partition == "nao" or partition == "iao":
        return partition
    else:
        raise NameError(" | Invalid partition scheme")


def mapping(arr, perm):
    """
    Maps the elements of a list according to a permutation.
    Args:
        arr: List of elements.
        perm: Permutation of the elements.

    Returns:
        List of elements corresponding to a given permutation.
    """
    return [arr[i] for i in perm]


def get_natorbs(mf, S):
    """
    Obtains the natural orbitals from the SCF calculation.
    Args:
        mf: PySCF SCF object.
        S: Overlap matrix in an AO basis.

    Returns:
        List containing the occupancies and the Natural Orbitals.
    """
    from scipy.linalg import eigh
    import numpy as np
    rdm1 = mf.make_rdm1(ao_repr=True)  # In AO basis
    # Only Restricted orbitals are supported with current version
    if np.ndim(rdm1) == 3:
        D = rdm1[0] + rdm1[1]
    elif np.ndim(rdm1) == 2:
        D = rdm1
    else:
        raise ValueError(" | Could not find dimensions for the 1-RDM")

    occ, coeff = eigh(np.linalg.multi_dot((S, D, S)), b=S)
    coeff = coeff[:, ::-1]  # Order coefficients
    occ = occ[::-1]  # Order occupancies
    occ[occ < 10 ** -12] = 0.0  # Set small occupancies to 0
    occ = np.diag(occ)
    return occ, coeff


def build_eta(mol):
    """
    Builds the eta matrices for the partitioning. They consist on a block-truncated matrix with all elements being
    zero except for the diagonal elements corresponding to the basis functions of a given atom, which are set to one.
    Args:
        mol: PySCF Mole object.

    Returns:
        List containing the eta matrices for each atom.
    """
    eta = [np.zeros((mol.nao, mol.nao)) for i in range(mol.natm)]
    for i in range(mol.natm):
        start = mol.aoslice_by_atom()[i, -2]
        end = mol.aoslice_by_atom()[i, -1]
        eta[i][start:end, start:end] = np.eye(end - start)
    return eta

def build_connec(Smo, thres=0.3):
    if wf_type(Smo) == 'rest':
        natoms = len(Smo)
        factor = 2
    else:
        natoms = len(Smo[0])
        factor = 1
    connectivity_matrix = np.zeros((natoms, natoms))

    for i in range(1, natoms + 1):
        for j in range(1, natoms + 1):
            if i != j:
                if factor * find_di(Smo, i, j) >= thres:
                    connectivity_matrix[i - 1, j - 1] = 1
    return connectivity_matrix

def build_connec_no(Smo, thres=0.3):
    natoms = len(Smo[0])
    connectivity_matrix = np.zeros((natoms, natoms))

    for i in range(1, natoms + 1):
        for j in range(1, natoms + 1):
            if i != j:
                if find_di_no(Smo, i, j) >= thres:
                    connectivity_matrix[i - 1, j - 1] = 1
    return connectivity_matrix

def find_rings(connec, minlen=6, maxlen=6):
    def dfs(connec, minlen, maxlen, start):
        stack = [(start, [start])]
        while stack:
            (v, path) = stack.pop()
            if len(path) > maxlen + 1:
                continue
            if len(path) >= minlen and connec[path[-1]][path[0]] == 1 and path[1] < path[-1] and path[0] == start:  # Check for circuit formation
                yield [p + 1 for p in path]
            for next in range(len(connec[v])):
                if connec[v][next] == 1 and next not in path:
                    stack.append((next, path + [next]))

    def is_unique_permutation(path, all_paths):
        for shift in range(len(path)):
            rot = np.roll(path, shift).tolist()
            if rot in all_paths or rot[::-1] in all_paths:
                return False
        return True

    all_paths = []
    starts = find_middle_nodes(connec)
    if not starts:
        starts = [0]
    for start in starts:
        for path in dfs(connec, minlen, maxlen, start):
            if is_unique_permutation(path, all_paths):
                all_paths.append(path)
    return all_paths

def filter_connec(connec):
    row_sums = np.sum(connec, axis=1)
    col_sums = np.sum(connec, axis=0)
    keep = np.where((row_sums > 1) & (col_sums > 1))[0]
    connec2 = connec[np.ix_(keep, keep)]
    return connec2

def is_onering(Smo):
    if wf_type(Smo) == "rest" or wf_type(Smo) == "unrest":
        connec = build_connec(Smo)
    else:
        connec = build_connec_no(Smo)
    filt = filter_connec(connec)
    print(filt)
    if np.any(np.sum(connec, axis=0) == 2):
        return True
    else:
        return False

def find_middle_nodes(connec2):
    row_sums = np.sum(connec2, axis=1)
    col_sums = np.sum(connec2, axis=0)
    keep = np.where((row_sums > 1) & (col_sums > 1))[0]
    middle_nodes = []
    for i in range(len(connec2)):  # Iterate over nodes in the subgraph
        if np.sum(connec2[i]) > 2:  # Check connections within the subgraph
            middle_nodes.append(keep[i])  # Get original node index
    return middle_nodes
