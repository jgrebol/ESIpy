from os import environ
import numpy as np
from collections import deque, defaultdict

environ["NUMEXPR_NUM_THREADS"] = "1"
environ["OMP_NUM_THREADS"] = "1"
environ["MKL_NUM_THREADS"] = "1"

def wf_type(aom):
    """
    Checks the topology of the AOMs to obtain the type of wavefunction.

    :param aom: The Atomic Overlap Matrices (AOMs) in the MO basis.
    :type aom: list of matrices
    :returns: A string with the type of wave function ('rest', 'unrest' or 'no').
    :rtype: str
    """
    if isinstance(aom[0][0][0], float):
        return "rest"
    elif isinstance(aom[1][0][0], float):
        return "no"
    elif isinstance(aom[1][0][0][0], float):
        return "unrest"
    else:
        raise NameError("Could not find the type of wave function from the AOMs")


def find_multiplicity(aom):
    """
    Checks the topology of the AOMs to get the difference between alpha and beta electrons.

    :param aom: The Atomic Overlap Matrices (AOMs) in the MO basis.
    :type aom: list of matrices
    :returns: A string with the multiplicity of the calculations.
    :rtype: str
    """

    if len(aom[0][0]) == len(aom[1][0]):
        return "singlet"
    elif len(aom[0][0]) == (len(aom[1][0]) + 1):
        return "doublet"
    elif len(aom[0][0]) == (len(aom[1][0]) + 2):
        return "triplet"
    elif len(aom[0][0]) == (len(aom[1][0]) + 3):
        return "quadruplet"
    elif len(aom[0][0]) == (len(aom[1][0]) + 4):
        return "quintuplet"
    else:
        return None


def load_file(file):
    """
    Loads a variable from a file.

    :param file: Contains the name or the path containing the variable.
    :type file: str
    :returns: The file loaded.
    :rtype: object
    """

    from pickle import load
    with open(file, "rb") as f:
        file = load(f)
    return file


def save_file(file, save):
    """
    Saves a variable to a file.

    :param file: The variable to be stored.
    :type file: object
    :param save: The name or the path where the variable will be saved.
    :type save: str
    :returns: None
    """

    from pickle import dump
    with open(save, "wb") as f:
        dump(file, f)


def find_distances(arr, geom):
    """
    Collects the distance between the atoms in ring connectivity.

    :param arr: Indices of the atoms in ring connectivity.
    :type arr: list of int
    :param geom: Geometry of the molecule as in mol.atom_coords().
    :type geom: numpy.ndarray
    :returns: List containing the distances of the members of the ring in Bohrs.
    :rtype: list of float
    """

    distances = []
    for i in range(len(arr)):
        coord1 = geom[arr[i] - 1]
        coord2 = geom[arr[(i + 1) % len(arr)] - 1]
        distances.append(np.linalg.norm(coord1 - coord2) * 0.529177249)
    return distances


def find_dis(arr, aom):
    """
    Collects the DIs between the atoms in ring connectivity.

    :param arr: Indices of the atoms in ring connectivity.
    :type arr: list of int
    :param aom: The Atomic Overlap Matrices (AOMs) in the MO basis.
    :type aom: list of matrices
    :returns: List containing the DIs of the members of the ring.
    :rtype: list of float
    """

    return [4 * np.trace(np.dot(aom[arr[i] - 1], aom[arr[(i + 1) % len(arr)] - 1])) for i in range(len(arr))]


def find_di(aom, i, j):
    """
    Collects the DI between two atoms.

    :param aom: The Atomic Overlap Matrices (AOMs) in the MO basis.
    :type aom: list of matrices
    :param i: Index of the first atom.
    :type i: int
    :param j: Index of the second atom.
    :type j: int
    :returns: DI between the atoms i and j.
    :rtype: float
    """

    return 2 * np.trace(np.dot(aom[i - 1], aom[j - 1]))


def find_di_no(aom, i, j):
    """
    Collects the DI between two atoms for correlated wavefunctions.

    :param aom: The Atomic Overlap Matrices (AOMs) in the MO basis.
    :type aom: list of matrices
    :param i: Index of the first atom.
    :type i: int
    :param j: Index of the second atom.
    :type j: int
    :returns: DI between the atoms i and j.
    :rtype: float
    """

    return np.trace(np.linalg.multi_dot((aom[1], aom[0][i - 1], aom[1], aom[0][j - 1])))


def find_lis(arr, aom):
    """
    Collects the LIs between the atoms in ring connectivity.

    :param arr: Indices of the atoms in ring connectivity.
    :type arr: list of int
    :param aom: The Atomic Overlap Matrices (AOMs) in the MO basis.
    :type aom: list of matrices
    :returns: List containing the DIs of the members of the ring.
    :rtype: list of float
    """

    return [2 * np.trace(np.dot(aom[arr[i] - 1], aom[arr[i] - 1])) for i in range(len(arr))]


def find_ns(arr, aom):
    """
    Collects the atomic populations of all the atoms in the ring.

    :param arr: Indices of the atoms in ring connectivity.
    :type arr: list of int
    :param aom: The Atomic Overlap Matrices (AOMs) in the MO basis.
    :type aom: list of matrices
    :returns: List containing the atomic populations of the members of the ring.
    :rtype: list of float
    """

    return [2 * np.trace(aom[arr[i] - 1]) for i in range(len(arr))]


def av1245_pairs(arr):
    """
    Collects the series of atoms that fulfill the 1-2-4-5 relationship for the AV1245 calculation.

    :param arr: Indices of the atoms in ring connectivity.
    :type arr: list of int
    :returns: List containing the series of atoms that fulfill the 1-2-4-5 relationship.
    :rtype: list of tuple
    """

    return [(arr[i % len(arr)], arr[(i + 1) % len(arr)], arr[(i + 3) % len(arr)], arr[(i + 4) % len(arr)])
            for i in range(len(arr))]


def mol_info(mol=None, mf=None, save=None, partition=None, connec=None):
    """
    Obtains information from the molecule and the calculation to complement the main code function without requiring the 'mol' and 'mf' objects.

    :param mol: PySCF Mole object.
    :type mol: pyscf.gto.Mole
    :param mf: PySCF SCF object.
    :type mf: pyscf.scf.hf.SCF
    :param save: String with the name of the file to save the information.
    :type save: str
    :param partition: String with the name of the partition.
    :type partition: str
    :returns: Dictionary with the information of the molecule and the calculation.
    :rtype: dict
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

    if connec:
        info.update({"connec": connec})

    if save:
        save_file(info, save)

    return info


def build_connectivity(mat=None, threshold=None):
    """
    Build the connectivity dictionary based on the given atomic overlap matrices.

    Parameters:
        mat: Atomic overlap matrices. If None, uses self.aom or builds meta_lowdin AOMs
        threshold: Threshold for connectivity determination. If None, uses self.rings_thres

    Returns:
        Dictionary representing atomic connectivity
    """
    wf = wf_type(mat)

    if wf == "rest":
        graph = build_connec_rest(mat, threshold)
    elif wf == "unrest":
        graph = build_connec_unrest(mat, threshold)
    elif wf == "no":
        graph = build_connec_no(mat, threshold)
    else:
        graph = None

    return graph

# Add this function to `esipy/tools.py`
def process_fragments(aom, rings, done=False):
    """
    Processes fragments by combining AOMs and updating rings.

    :param aom: List of AOMs.
    :param rings: List of rings (can include sets for fragments).
    :returns: Updated AOMs, rings, and fragment AOMs.
    :rtype: tuple (list, list, list)
    """
    import numpy as np

    fragaom = []
    fragmap = {}
    nfrags = 0
    for ring in rings:
        for r in ring:
            if isinstance(r, set):
                if tuple(r) not in fragmap:
                    fragmap[tuple(r)] = nfrags + len(aom) + 1
                nfrags += 1
                if not done:
                    print(f" | Fragment FF{len(aom) + nfrags}: {r}")
                combined_aom = np.zeros_like(aom[0])
                for atm in r:
                    combined_aom += aom[atm - 1]
                fragaom.append(combined_aom)
            else:
                continue
    return fragaom, fragmap

def format_partition(partition):
    """
    Filters the 'partition' attribute for flexibility.

    :param partition: String with the name of the partition.
    :type partition: str
    :returns: String with the standard partition name for ESIpy.
    :rtype: str
    """

    partition = partition.lower()
    if partition in ["m", "mul", "mull", "mulliken"]:
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
    elif partition in ["iao-autosad", "autosad", "iaoa", "iao-a", "ia"]:
        return "iao-autosad"
    elif partition in ["q", "qt", "qtaim", "quant", "quantum"]:
        return "qtaim"
    else:
        raise NameError(" | Invalid partition scheme")


def format_short_partition(partition):
    """
    Filters the short version of the 'partition' attribute.

    :param partition: String with the name of the partition.
    :type partition: str
    :returns: String with the short version of the partition scheme.
    :rtype: str
    """

    partition = partition.lower()
    if partition == "mulliken":
        return "mul"
    elif partition == "lowdin":
        return "low"
    elif partition == "meta_lowdin":
        return "metalow"
    elif partition == "nao" or partition == "iao" or partition == "qtaim":
        return partition
    else:
        raise NameError(" | Invalid partition scheme")


def mapping(arr, perm):
    """
    Maps the elements of a list according to a permutation.

    :param arr: List of elements.
    :type arr: list
    :param perm: Permutation of the elements.
    :type perm: list of int
    :returns: List of elements corresponding to a given permutation.
    :rtype: list
    """

    return [arr[i] for i in range(len(perm))]


def get_natorbs(mf, S):
    """
    Obtains the natural orbitals from the SCF calculation.

    :param mf: PySCF SCF object.
    :type mf: pyscf.scf.hf.SCF
    :param S: Overlap matrix in an AO basis.
    :type S: numpy.ndarray
    :returns: List containing the occupancies and the Natural Orbitals.
    :rtype: tuple of (numpy.ndarray, numpy.ndarray)
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
    Builds the eta matrices for the partitioning. They consist of a block-truncated matrix with all elements being
    zero except for the diagonal elements corresponding to the basis functions of a given atom, which are set to one.

    :param mol: PySCF Mole object.
    :type mol: pyscf.gto.Mole
    :returns: List containing the eta matrices for each atom.
    :rtype: list of numpy.ndarray
    """
    import numpy as np
    eta = [np.zeros((mol.nao, mol.nao)) for _ in range(mol.natm)]
    for i in range(mol.natm):
        start = mol.aoslice_by_atom()[i, -2]
        end = mol.aoslice_by_atom()[i, -1]
        eta[i][start:end, start:end] = np.eye(end - start)
    return eta

def build_connec_rest(Smo, thres=0.25):
    natoms = len(Smo)
    connec_dict = {}

    for i in range(1, natoms + 1):
        for j in range(1, natoms + 1):
            if i != j:
                if 2 * find_di(Smo, i, j) >= thres:
                    if i not in connec_dict:
                        connec_dict[i] = []
                    connec_dict[i].append(j)
    return filter_connec(connec_dict)

def build_connec_unrest(Smo, thres=0.25):
    natoms = len(Smo[0])
    connec_dict = {}

    for i in range(1, natoms + 1):
        for j in range(1, natoms + 1):
            if i != j:
                di_alpha = find_di(Smo[0], i, j)
                di_beta = find_di(Smo[1], i, j)
                di = di_alpha + di_beta
                if di >= thres:
                    if i not in connec_dict:
                        connec_dict[i] = []
                    connec_dict[i].append(j)
    return filter_connec(connec_dict)

def build_connec_no(Smo, thres=0.25):
    connec_dict = {}

    for i in range(len(Smo[0])):
        for j in range(len(Smo[0])):
            if i != j:
                if find_di_no(Smo, i, j) >= thres:
                    if i not in connec_dict:
                        connec_dict[i] = []
                    connec_dict[i].append(j)
    return filter_connec(connec_dict)

def find_rings(connec, minlen=6, maxlen=6):
    def dfs(connec, minlen, maxlen, start):
        stack = [(start, [start])]
        while stack:
            (v, path) = stack.pop()
            if len(path) > maxlen:
                continue
            if len(path) >= minlen and start in connec[path[-1]] and path[1] < path[-1] and path[0] == start:
                yield path
            for next in connec.get(v, []):
                if next not in path:
                    stack.append((next, path.copy() + [next]))

    def unique(path, all):
        for shift in range(len(path)):
            rot = np.roll(path.copy(), shift).tolist()
            if rot in all or rot[::-1] in all:
                return False
        return True

    all_paths = []
    starts = find_middle_nodes(connec)
    if not starts:
        starts = [1]
    for start in starts:
        for path in dfs(connec, minlen, maxlen, start):
            if unique(path, all_paths):
                all_paths.append(path)
    return all_paths

def filter_connec(connec):
    filtered_connec = {}
    for key, values in connec.items():
        if len(values) > 1:
            filtered_vals = [v for v in values if len(connec[v]) > 1]
            if filtered_vals:
                filtered_connec[key] = filtered_vals
    return filtered_connec

def is_fused(arr, connec):
    for i in range(len(arr) - 1):
        if len([x for x in connec[arr[i]] if x in arr]) >= 3:
            return True
    return False

def find_middle_nodes(connec2):
    return [key for key, vals in connec2.items() if len(vals) > 2]

def find_node_distances(connec):
   distances = defaultdict(dict)

   for start in connec:
       queue = deque([(start, 0)])
       visited = set()

       while queue:
           cur_node, cur_dist = queue.popleft()

           if cur_node not in visited:
               visited.add(cur_node)
               distances[start][cur_node] = cur_dist

               for neighbor in connec[cur_node]:
                   if neighbor not in visited:
                       queue.append((neighbor, cur_dist + 1))

   return distances


def get_efos2(mol, mf):
    from pyscf.lo.nao import _prenao_sub
    from pyscf import gto, scf, dft

    aufbau_order = {
        '1s': 1, '2s': 2, '2p': 3, '3s': 4, '3p': 5, '4s': 6,
        '3d': 7, '4p': 8, '5s': 9, '4d': 10, '5p': 11, '6s': 12,
        '4f': 13, '5d': 14, '6p': 15, '7s': 16, '5f': 17, '6d': 18,
        '7p': 19
    }

    unpaired = {
        "H": 1, "He": 0, "Li": 1, "Be": 0, "B": 1, "C": 2, "N": 3, "O": 2, "F": 1, "Ne": 0,
        "Na": 1, "Mg": 0, "Al": 1, "Si": 2, "P": 3, "S": 2, "Cl": 1, "Ar": 0, "K": 1, "Ca": 0,
    }

    minorbs = {
        "H": 1, "He": 1, "Li": 2, "Be": 2, "B": 5, "C": 5, "N": 5, "O": 5, "F": 5, "Ne": 5,
        "Na": 6, "Mg": 6, "Al": 9, "Si": 9, "P": 9, "S": 9, "Cl": 9, "Ar": 9, "K": 10, "Ca": 10,
    }

    aoslices = mol.aoslice_by_atom()

    # Get the unique atom symbols in the molecule
    atoms = []
    atom_indices = []
    for ia in range(mol.natm):
        symb = mol.atom_pure_symbol(ia)
        if symb not in atoms:
            atoms.append(symb)
            atom_indices.append(ia)

    # Sort the atom indices based on the order of appearance in the molecule

    atom_veps = {}
    atom_vaps = {}
    for atom_symbol, ia in zip(atoms, atom_indices):
        start, end = aoslices[ia, 2], aoslices[ia, 3]

        # Create a new molecule object for the current atom
        molatm = gto.Mole()
        molatm.atom = f"{atom_symbol} 0.0 0.0 0.0"
        molatm.basis = {atom_symbol: mol._basis[atom_symbol]}
        molatm.spin = unpaired[atom_symbol]
        molatm.charge = 0
        mol.symmetry = False
        molatm.build()

        orbital_labels = [label[2] for label in molatm.ao_labels(fmt=False)]

        # Create a new SCF object for the atom
        if "HF" in mf.__class__.__name__:
            mfatm = scf.HF(molatm)
        elif "KS" in mf.__class__.__name__:
            mfatm = scf.KS(molatm)
        else:
            mfatm = mf.__class__(molatm)

        mfatm.kernel()

        # Build AO 1-RDM manually
        if "U" in mfatm.__class__.__name__:
            Ca, Cb = mfatm.mo_coeff
            occa, occb = mfatm.mo_occ
            Da = np.einsum('pi,i,qi->pq', Ca, occa, Ca)
            Db = np.einsum('pi,i,qi->pq', Cb, occb, Cb)
            Patm = Da + Db
        else:
            C = mfatm.mo_coeff
            occ = mfatm.mo_occ
            Patm = np.einsum('pi,i,qi->pq', C, occ, C)

        Satm = mfatm.get_ovlp()
        PSatm = np.dot(Patm, Satm)
        SPSatm = np.dot(Satm, PSatm)

        vaps_atom, veps_atom = _prenao_sub(molatm, SPSatm, Satm)

        # Order the orbitals based on Aufbau's principle
        atom_orbitals = orbital_labels[start:end]
        aufbau_values = [aufbau_order.get(orb, float('inf')) for orb in orbital_labels]

        ordered_indices = np.argsort([-val for val in aufbau_values])

        # Only keep the first n_occ occupied orbitals for each atom
        n_occ = minorbs[atom_symbol]  # or use your own dictionary for occupied orbitals
        vaps_ordered = vaps_atom[ordered_indices][::-1][:n_occ]
        veps_ordered = veps_atom[:, ordered_indices][:, ::-1][:, :n_occ]
        print(np.shape(veps_ordered))

        atom_veps[atom_symbol] = veps_ordered
        atom_vaps[atom_symbol] = vaps_ordered

    aoslices = mol.aoslice_by_atom()
    nminbas = sum(atom_veps[mol.atom_pure_symbol(ia)].shape[1] for ia in range(mol.natm))
    veps_block = np.zeros((mol.nao, nminbas))
    vaps_diag = []

    col = 0
    for ia in range(mol.natm):
        atom_symbol = mol.atom_pure_symbol(ia)
        start, end = aoslices[ia, 2], aoslices[ia, 3]
        veps = atom_veps[atom_symbol]
        vaps = atom_vaps[atom_symbol]
        ncol = veps.shape[1]
        veps_block[start:end, col:col + ncol] = veps
        vaps_diag.extend(list(vaps))
        #vaps_diag[start:end, col:col + ncol] = np.diag(vaps)
        col += ncol

    np.set_printoptions(threshold=np.inf, linewidth=200, suppress=True)

    return vaps_diag, veps_block

def autosad(mol, mf):
    import numpy as np
    from pyscf import scf, lo

    # Getting data
    S1 = mol.intor_symmetric('int1e_ovlp')  # In AO basis
    C = get_efos2(mol, mf)[1] # (nbas×nminbas)
    C_occ = mf.mo_coeff[:, :mol.nelectron // 2]

    # Overlaps respecte la base minima
    S12 = S1 @ C  # (nbas×nminbas)
    S2 = C.T @ S1 @ C  # Linear combination of original AOs

    # Building the projectors
    S21 = S12.T
    Ctild_min = np.linalg.solve(S2, S21 @ C_occ)  # shape (n_min×n_occ)

    try:
        P12 = np.linalg.solve(S1, S12)  # S1 x P12 = S12
        Ctild_AO = np.linalg.solve(S1, S12 @ Ctild_min) # Ctild_AO = P12 C_tild_min
    except np.linalg.LinAlgError:
        # S1 is ill-conditioned: use canonical orthonormalization as in PySCF
        X = scf.addons.canonical_orth_(S1, lindep_threshold=1e-8)
        P12 = X @ X.T @ S12
        Ctild_AO = P12 @ Ctild_min

    # Ortonormalitzar els orbitals projectats
    Ctild = lo.vec_lowdin(Ctild_AO, S1)  # W(WSW)**(-1/2)

    # Mes projectors
    P_occ = C_occ @ C_occ.T @ S1  # projector onto occupied space (equiv a CCS1)
    P_proj = Ctild @ Ctild.T @ S1  # projector onto projected space throuhg polarized orbitals (equiv a CCS2)
    IAOs = P12 + 2 * (P_occ @ P_proj @ P12) - P_occ @ P12 - P_proj @ P12 # PySCF expression

    return IAOs

