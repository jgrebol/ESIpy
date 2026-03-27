import os
from collections import deque, defaultdict
import numpy as np

def wf_type(aom):
    """
    Checks the topology of the AOMs to obtain the type of wavefunction.

    :param aom: The Atomic Overlap Matrices (AOMs) in the MO basis.
    :returns: A string with the type of wave function ('rest', 'unrest' or 'no').
    """
    # Restricted: list of 2D matrices
    if isinstance(aom[0], np.ndarray) and aom[0].ndim == 2:
        return "rest"
    
    # Unrestricted or Natural Orbitals: [list, list/array]
    if isinstance(aom, list) and len(aom) == 2:
        # Natural Orbitals: [list of matrices, 1D/2D array of occupations]
        if isinstance(aom[1], np.ndarray) and aom[1].ndim in [1, 2]:
            return "no"
        # Unrestricted: [list of alpha matrices, list of beta matrices]
        if isinstance(aom[1], list) and isinstance(aom[1][0], np.ndarray) and aom[1][0].ndim == 2:
            return "unrest"
            
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

    return [4 * np.einsum('ij,ji->', aom[arr[i] - 1], aom[arr[(i + 1) % len(arr)] - 1]) for i in range(len(arr))]


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

    return 2 * np.einsum('ij,ji->', aom[i - 1], aom[j - 1])


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

    # Tr(Occ @ AOM_i @ Occ @ AOM_j)
    occ = np.diag(aom[1])
    return np.einsum('ij,jk,kl,li->', occ, aom[0][i - 1], occ, aom[0][j - 1])


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

    return [2 * np.einsum('ij,ji->', aom[arr[i] - 1], aom[arr[i] - 1]) for i in range(len(arr))]


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


def mol_info(mol=None, mf=None, save=None, partition=None, connec=None, iaoref=None, iaopol=None, iaomix=None, heavy_only=True, full_basis=False):
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
    :param iaoref: Reference basis for IAO.
    :type iaoref: str
    :param iaopol: Polarization basis for IAO.
    :type iaopol: str
    :param iaomix: IAO mixing weight.
    :type iaomix: float
    :returns: Dictionary with the information of the molecule and the calculation.
    :rtype: dict
    """

    info = {}
    info.update({"partition": partition})
    if iaoref: info.update({"iaoref": iaoref})
    if iaopol: info.update({"iaopol": iaopol})
    if iaomix is not None: info.update({"iaomix": iaomix})
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

    # Include connectivity if a value was passed
    if connec is not None:
        info.update({"connec": connec})

    if save:
        save_file(info, save)

    return info


def build_connectivity(mat=None, threshold=0.25):
    """
    Build the connectivity dictionary based on the given atomic overlap matrices.

    Parameters:
        mat: Atomic overlap matrices. If None, uses self.aom or builds meta-lowdin AOMs
        threshold: Threshold for connectivity determination. Default is 0.25.

    Returns:
        Dictionary representing atomic connectivity
    """
    if threshold is None: threshold = 0.25
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
                t_r = tuple(sorted(list(r)))
                if t_r not in fragmap:
                    nfrags += 1
                    fragmap[t_r] = nfrags + len(aom)
                    if not done:
                        print(f" | Fragment FF{len(aom) + nfrags}: {r}")
                    combined_aom = np.zeros_like(aom[0])
                    for atm in r:
                        combined_aom += aom[atm - 1]
                    fragaom.append(combined_aom)
            else:
                continue
    return fragaom, fragmap



def format_partition(partition, iaoref='minao', iaopol=None, iaomix=None, heavy_only=False):
    import re
    orig = partition
    p_split = partition.split(None, 1)
    p_method = p_split[0].lower()
    p_suffix = p_split[1] if len(p_split) > 1 else ""
    
    # Guidelines: IAO <basis>, FPIAO(x) <basis>, DFPIAO(x) <basis>
    
    # 1. Standardize method name
    if p_method in ["m", "mul", "mull", "mulliken"]: base = "mulliken"
    elif p_method in ["l", "low", "lowdin"]: base = "lowdin"
    elif p_method in ["ml", "mlow", "m-low", "meta-low", "metalow", "mlowdin", "m-lowdin", "metalowdin", "meta-lowdin", "meta-lowdin"]: base = "meta-lowdin"
    elif p_method in ["n", "nao", "natural", "nat"]: base = "nao"
    elif p_method in ["i", "iao", "intrinsic", "intr"]: base = "iao"
    elif p_method in ["iao-autosad", "iaoauto", "iaoa", "iaa", "ia", "a", "autosad", "iaosad", "autos"]: base = "iao-autosad"
    elif p_method in ["iao-effao-gross", "iao-eg", "iaoeg", "iaog", "ig", "gross", "iag", "g"]: base = "iao-effao-gross"
    elif p_method in ["iao-effao-net", "iao-en", "iaoen", "iaon", "in", "net", "ian", "ne"]: base = "iao-effao-net"
    elif p_method in ["iao-effao-lowdin", "iaoel", "iaol", "il", "iel", "iae"]: base = "iao-effao-lowdin"
    elif p_method in ["iao-effao-metalowdin", "iao-effao-meta-lowdin", "iaom", "im"]: base = "iao-effao-metalowdin"
    elif p_method in ["sym", "ias", "is", "iao-effao-symmetric"]: base = "iao-effao-symmetric"
    elif p_method in ["sps", "iao-effao-sps"]: base = "iao-effao-sps"
    elif p_method in ["spsa", "iao-effao-spsa"]: base = "iao-effao-spsa"
    elif p_method == "iao_basis": base = "iao-basis"
    elif p_method == "fpiao": base = "fpiao"
    elif p_method == "dfpiao": base = "dfpiao"
    elif p_method == "xiao_dfpiao": base = "xiao_dfpiao"
    else: base = p_method.lower()
    
    # 2. Extract weight if present: "dfpiao(0.3)" -> "dfpiao", weight=0.3
    match_w = re.search(r"\(+(.*?)\)+", partition)
    if match_w:
        try:
            weight = float(match_w.group(1))
            # Clean base name if weight was part of it
            base = re.sub(r"\(.*?\)", "", base).strip()
        except:
            weight = None
    else:
        weight = None
        
    if weight is None:
        if iaomix is not None:
            weight = iaomix if isinstance(iaomix, (float, int)) else (iaomix[0] if iaomix else 0.5)
        else:
            if "fpiao" in p_method: weight = 1.0
            elif "dfpiao" in p_method: weight = 0.5
            else: weight = 0.5

    # 3. Extract basis
    # Basis can be in p_suffix or iaoref
    res_basis = p_suffix.strip()
    if not res_basis:
        res_basis = iaoref if iaoref else ""
    
    # Clean basis name (remove extension, take filename)
    if "/" in res_basis:
        res_basis = res_basis.split("/")[-1].replace("_ref_basis.dat", "").replace("_polar_basis.dat", "").lower()
    elif res_basis.lower() == "minao":
        res_basis = ""
    else:
        res_basis = res_basis.lower()

    # 4. Construct final label
    if "fpiao" in base or "dfpiao" in base or "xiao" in base:
        # Avoid double dots like 1.00
        w_str = f"{weight:g}" if weight != int(weight) else f"{weight:.1f}"
        if "(" not in base:
            base += f"({w_str})"
        else:
            base = re.sub(r"\(.*?\)", f"({w_str})", base)

    label = base
    # Only append reference basis for IAO variants
    is_iao = "iao" in base or "piao" in base or "xiao" in base
    if is_iao and res_basis:
        label += f" {res_basis}"
        
    return label.lower()
def format_short_partition(partition):

    # Split to preserve case for paths/basis names if needed
    p_split = partition.split(None, 1)
    p_method = p_split[0].lower()
    p_suffix = " " + p_split[1] if len(p_split) > 1 else ""
    partition = p_method + p_suffix

    if partition == "mulliken":
        return "mul"
    elif partition == "lowdin":
        return "low"
    elif partition == "meta-lowdin":
        return "metalow"
    elif partition == "iao-effao-lowdin":
        return "iao-effao-low"
    elif partition == "iao-effao-gross":
        return "iao-effao-gross"
    elif partition == "iao-effao-net":
        return "iao-effao-net"
    elif partition == "iao-ano":
        return "iano"
    elif partition == "piao":
        return "p"
    elif partition == "piao-iao":
        return "pi"
    elif partition == "piao-iao-ano":
        return "pia"
    elif partition in ("nao", "iao", "qtaim", "iao-autosad", "iao-effao"):
        return partition
    else:
        return partition



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
    print(" | Obtaining Natural Orbitals from the 1-RDM...")

    rdm1 = mf.make_rdm1(ao_repr=True)
    rdm1_arr = np.asarray(rdm1)

    if rdm1_arr.ndim == 3:
        raise NotImplementedError(" | Natural Orbitals for unrestricted calculations are not implemented yet.")
        #D = np.sum(rdm1_arr, axis=0)
    elif rdm1_arr.ndim == 2:
        D = rdm1_arr
    else:
        raise ValueError(" | Could not find dimensions for the 1-RDM")

    # Ensure S is symmetric, which by default should be
    S = np.asarray(S, dtype=float)
    S = 0.5 * (S + S.T)

    # Solve generalized eigenproblem to obtain NOs and occupancies
    occ, coeff = eigh(np.linalg.multi_dot((S, D, S)), b=S)
    print(f"DEBUG: eigh eigenvalues (occupancies): {occ}")

    # Order descending
    coeff = coeff[:, ::-1]
    occ = occ[::-1]
    occ[occ < 1e-12] = 0.0 # Small values set to zero
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
    connec_dict = {i: [] for i in range(1, natoms + 1)}

    for i in range(1, natoms):
        for j in range(i + 1, natoms + 1):
            if 2 * find_di(Smo, i, j) >= thres:
                connec_dict[i].append(j)
                connec_dict[j].append(i)
    return filter_connec({k: v for k, v in connec_dict.items() if v})

def build_connec_unrest(Smo, thres=0.25):
    natoms = len(Smo[0])
    connec_dict = {i: [] for i in range(1, natoms + 1)}

    for i in range(1, natoms):
        for j in range(i + 1, natoms + 1):
            di_alpha = find_di(Smo[0], i, j)
            di_beta = find_di(Smo[1], i, j)
            di = di_alpha + di_beta
            if di >= thres:
                connec_dict[i].append(j)
                connec_dict[j].append(i)
    return filter_connec({k: v for k, v in connec_dict.items() if v})

def build_connec_no(Smo, thres=0.25):
    natoms = len(Smo[0])
    connec_dict = {i: [] for i in range(1, natoms + 1)}

    for i in range(1, natoms):
        for j in range(i + 1, natoms + 1):
            if find_di_no(Smo, i, j) >= thres:
                connec_dict[i].append(j)
                connec_dict[j].append(i)
    return filter_connec({k: v for k, v in connec_dict.items() if v})

def find_rings(connec, minlen=6, maxlen=6, exclude=None):
    exclude = set(exclude) if exclude else set()
    def dfs(connec, minlen, maxlen, start):
        if start in exclude:
            return

        stack = [(start, [start])]
        while stack:
            (v, path) = stack.pop()
            if len(path) > maxlen:
                continue
            if len(path) >= minlen and start in connec[path[-1]] and path[1] < path[-1] and path[0] == start:
                yield path
            for next in connec.get(v, []):
                if next not in path and next not in exclude:
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
    if len(connec) < 2: # With two elements we can not filter
        return connec
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

try:
    from esipy.iao import iao, get_effaos, autosad
except ImportError:
    pass

def permute_aos_rows(mat, mole2):
    """
    Reorders FCHK AO rows to match PySCF's internal layout AND applies
    normalization scaling for Cartesian basis sets.
    """
    mol = mole2.pyscf_mol
    is_cart = bool(getattr(mol, 'cart', False))

    # Scalings for Cartesian basis functions
    pi = np.pi
    p1 = 2.0 * np.sqrt(pi / 15.0)
    p2 = 2.0 * np.sqrt(pi / 5.0)
    p3 = 2.0 * np.sqrt(pi / 7.0)
    p4 = 2.0 * np.sqrt(pi / 35.0)
    p5 = 2.0 * np.sqrt(pi / 105.0)
    p6 = (2.0 / 3.0) * np.sqrt(pi)
    p7 = (2.0 / 3.0) * np.sqrt(pi / 7.0)
    p8 = (2.0 / 3.0) * np.sqrt(pi / 35.0)
    p9 = 2.0 * np.sqrt(pi / 11.0)
    p10 = (2.0 / 3.0) * np.sqrt(pi / 11.0)
    p11 = 2.0 * np.sqrt(pi / 231.0)
    p12 = (2.0 / 3.0) * np.sqrt(pi / 77.0)
    p13 = 2.0 * np.sqrt(pi / 1155.0)

    SDIAG = {
        # 6D: XX, XY, XZ, YY, YZ, ZZ
        2: [p2, p1, p1, p2, p1, p2],

        # 10F: XXX, XXY, XXZ, XYY, XYZ, XZZ, YYY, YYZ, YZZ, ZZZ
        3: [p3, p4, p4, p4, p5, p4, p3, p4, p4, p3],

        # 15G: XXXX ... ZZZZ
        4: [p6, p7, p7, p5, p8, p5, p7, p8, p8, p7, p6, p7, p5, p7, p6],

        # 21H
        5: [p9, p10, p10, p11, p12, p11, p11, p13, p13, p11, p10, p12,
            p13, p12, p10, p9, p10, p11, p11, p10, p9]
    }

    # Mapping from FCHK to PySCF ordering
    MAPS = {
        # Cartesian
        2: [0, 3, 4, 1, 5, 2],
        3: [0, 4, 5, 3, 9, 6, 1, 8, 7, 2],
        4: [0, 4, 5, 3, 14, 6, 11, 13, 12, 9, 1, 8, 7, 10, 2],
        5: [0, 5, 6, 4, 20, 7, 15, 19, 18, 11, 10, 14, 13, 17, 16, 1, 9, 8, 12, 11, 2],
        6: [0, 6, 7, 5, 27, 8, 19, 26, 25, 13, 15, 22, 21, 24, 23, 10, 18, 17, 20, 19, 16, 1, 10, 9, 14, 13, 11, 2],

        # Spherical
        -2: [4, 2, 0, 1, 3],
        -3: [6, 4, 2, 0, 1, 3, 5],
        -4: [8, 6, 4, 2, 0, 1, 3, 5, 7],
        -5: [10, 8, 6, 4, 2, 0, 1, 3, 5, 7, 9],
        -6: [12, 10, 8, 6, 4, 2, 0, 1, 3, 5, 7, 9, 11]
    }

    atom_map = np.asarray(mole2.fchk_basis_arrays['iatsh']) - 1
    shell_types = np.asarray(mole2.fchk_basis_arrays['mssh'])

    registry = {}
    cursor = 0
    for iat, st in zip(atom_map, shell_types):
        if st == -1:
            subs = [(0, 1), (1, 3)]
        elif st >= 0:
            subs = [(st, (st + 1) * (st + 2) // 2 if st > 1 else (3 if st == 1 else 1))]
        else:
            l = abs(st)
            subs = [(l, 2 * l + 1)]

        for l, n in subs:
            key = (iat, l)
            if key not in registry: registry[key] = []
            registry[key].append({'start': cursor, 'count': n})
            cursor += n

    idx_list = []
    scale_list = []  # Stores 1.0 or 1.0/Sdiag for every AO
    usage = {}

    for b in range(mol.nbas):
        iat, l = int(mol.bas_atom(b)), int(mol.bas_angular(b))
        key = (iat, l)
        count = usage.get(key, 0)
        target = registry[key][count]
        usage[key] = count + 1

        if l <= 1:
            # S and P: Identity map, Scale = 1.0
            indices = [target['start'] + i for i in range(target['count'])]
            scales = [1.0] * target['count']
        else:
            # High-L
            lookup = l if is_cart else -l
            order = MAPS.get(lookup, list(range(target['count'])))
            indices = [target['start'] + i for i in order]

            # Apply Mokit Scaling if Cartesian and we have constants for it
            if is_cart and l in SDIAG:
                # Mokit: coeff = coeff / Sdiag
                # We implement multiplication by (1.0 / Sdiag)
                scales = [1.0 / s for s in SDIAG[l]]
            else:
                scales = [1.0] * target['count']

        idx_list.extend(indices)
        scale_list.extend(scales)

    p = np.array(idx_list)
    s = np.array(scale_list)

    if mat.ndim == 2:
        if mat.shape[0] == len(p):  # (NAO, NMO) - Permute rows
            return mat[p] * s[:, None]
        else:  # (NMO, NAO) - Permute cols
            return mat[:, p] * s[None, :]

    if mat.ndim == 3:  # (Spin, NAO, NMO) or similar
        if mat.shape[1] == len(p):
            return mat[:, p, :] * s[None, :, None]
        else:
            return mat[:, :, p] * s[None, None, :]

    return mat
