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
    import os
    from pickle import dump
    # Ensure parent directory exists
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

    if not connec:
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

def iao(mf_orig, mf_min, coeffs=None):
    """
    Build IAOs using a minimal basis (mf_min) and the original AO basis (mf_orig).

    mf_orig, mf_min:
        Objects containing:
        - nbasis, numprim
        - coefpb, expp, coord, iptoat, nlm
        - All data needed by build_cross_ovlp()

    mo_coeff:
        Molecular orbital coefficients in AO basis (nbas_orig × nmo)

    nelectron:
        Total number of electrons (for closed-shell, we take nocc = nelec//2)
    """

    # --- Step 1: Overlap matrices ---
    from pyscf.gto.mole import intor_cross
    S1 = mf_orig.get_ovlp()  # (nbas_orig × nbas_orig)
    S2 = mf_min.get_ovlp()   # (nbas_min × nbas_min)
    S12 = intor_cross("int1e_ovlp", mf_orig.mol, mf_min.mol)  # (nbas_orig × nbas_min)

    # --- Step 2: Occupied space ---
    nocc = mf_orig.nelec // 2
    if coeffs is None:
        coeffs = mf_min.mo_coeff  # (nbas_orig × nmo)
    C_occ = coeffs[:, :nocc]  # (nbas_orig × nocc)

    # --- Step 3: Build projectors ---
    S21 = S12.T
    Ctild_min = np.linalg.solve(S2, S21 @ C_occ)  # (nmin × nocc)

    try:
        P12 = np.linalg.solve(S1, S12)  # (nbas_orig × nbasis_min)
        Ctild_AO = np.linalg.solve(S1, S12 @ Ctild_min)
    except np.linalg.LinAlgError:
        from pyscf import scf
        # Fallback to canonical orthonormalization
        X = scf.addons.canonical_orth_(S1, lindep_threshold=1e-8)
        P12 = X @ X.T @ S12
        Ctild_AO = P12 @ Ctild_min

    # --- Step 4: Orthonormalize projected orbitals ---
    from pyscf.lo.orth import vec_lowdin
    Ctild = vec_lowdin(Ctild_AO, S1)

    # --- Step 5: PySCF-style IAO projector formula ---
    P_occ  = C_occ @ C_occ.T @ S1
    P_proj = Ctild @ Ctild.T @ S1
    IAOs = P12 + 2 * (P_occ @ P_proj @ P12) - P_occ @ P12 - P_proj @ P12

    # --- Step 6: Orthonormalize IAOs ---
    IAOs = vec_lowdin(IAOs, S1)

    return IAOs

# NOT IN USE YET
from pyscf.lo.nao import _sph_average_mat, _cart_average_mat, _core_val_ryd_list
from pyscf.lo.orth import weight_orth
from scipy.linalg import eigh
def get_shell_L_indices(mol):
    """
    Return a list of AO-index arrays grouped per-atom and per-angular-momentum L.

    For each atom, this function collects all shells (rows in mol._bas) belonging to
    that atom and for each angular momentum L it concatenates the AO indices of all
    shells with that L (this reproduces PySCF's grouping used by the spherical/cartesian
    averaging helpers). SP shells in `mol._bas` are already split into l=0 and l=1
    entries by `make_bas_env`, so they will be treated as separate L groups here.
    """
    import numpy as _np

    # Use mol._bas and mol.ao_loc_nr() (PySCF-style). This respects the actual AO ordering
    # used elsewhere in the code and naturally groups multiple contracted shells per L.
    bas = _np.asarray(getattr(mol, '_bas'))

    # bas columns: [ATOM_OF, ANG_OF, NPRIM_OF, NCTR_OF, KAPPA_OF, PTR_EXP, PTR_COEFF, UNUSED]
    atom_of = bas[:, 0].astype(int)
    ang_of = bas[:, 1].astype(int)

    # Build ao_loc from _bas to ensure consistency with contraction counts
    ao_loc_list = [0]
    for ish in range(len(bas)):
        ang = int(ang_of[ish])
        nctr = int(bas[ish, 3]) if bas.shape[1] > 3 else 1
        if nctr <= 0:
            nctr = 1
        # degeneracy per angular momentum (cartesian vs spherical)
        if hasattr(mol, 'cart') and mol.cart:
            degen_shell = (ang + 1) * (ang + 2) // 2
        else:
            degen_shell = 2 * ang + 1
        if degen_shell <= 0:
            degen_shell = 1
        ao_loc_list.append(ao_loc_list[-1] + nctr * degen_shell)
    ao_loc = _np.array(ao_loc_list, dtype=int)

    shell_idx_list = []
    # iterate atoms in order
    for ia in range(mol.natm):
        # find shell indices for this atom
        shells = [ish for ish in range(len(bas)) if atom_of[ish] == ia]
        if not shells:
            continue
        # collect L values present for this atom in ascending order
        Ls = sorted({int(ang_of[ish]) for ish in shells})
        for L in Ls:
            # collect AO indices from all shells on this atom with angular momentum L
            idx = []
            for ish in shells:
                if int(ang_of[ish]) != L:
                    continue
                start = int(ao_loc[ish])
                end = int(ao_loc[ish + 1])
                idx.extend(range(start, end))
            shell_idx_list.append(_np.array(idx, dtype=int))

    return shell_idx_list

def get_shell_L_indices_fchk(mol):
    """
    Build AO-index groups per-atom and angular momentum L following the original
    Gaussian FCHK shell ordering (i.e., using `mol.mssh`/`mol.mnsh`/`mol.iatsh`).

    Returns a list of numpy arrays where each array is the AO indices (in FCHK
    AO ordering) corresponding to one atomic-L block (same semantics as
    PySCF's grouping but using FCHK ordering).
    """
    import numpy as _np

    # Number of shells and basic arrays from the FCHK reader
    shell_types = [int(x) for x in mol.mssh]
    mnsh = [int(x) for x in mol.mnsh]
    shell_atom = _np.array([int(x) for x in mol.iatsh], dtype=int)
    natm = int(mol.natm)

    # Convert 1-based atom indices (common in FCHK) to 0-based
    if shell_atom.min() == 1 and shell_atom.max() <= natm:
        shell_atom = shell_atom - 1

    # map shell_type -> number of AOs contributed by that contracted shell
    mult_map = {0: 1, 1: 3, 2: 6, 3: 10, 4: 15, 5: 21,
                -1: 4, -2: 5, -3: 7, -4: 9, -5: 11}

    nbas = len(shell_types)

    # Build primitive pointer (primitives per shell) like earlier functions
    prim_ptr = [0]
    for n in mnsh:
        prim_ptr.append(prim_ptr[-1] + n)

    # Build AO start offsets per shell following FCHK ordering
    ao_loc_shell = [0]
    for ish in range(nbas):
        st = shell_types[ish]
        # number of primitives in this contracted shell
        p0 = prim_ptr[ish]
        p1 = prim_ptr[ish + 1]
        nprim = p1 - p0 if (p1 > p0) else 1

        if st == -1:
            # SP shell: S coefficients in c1, P in c2 (if present)
            try:
                nctr_s = (len(mol.c1[p0:p1]) // nprim) if hasattr(mol, 'c1') and mol.c1 is not None else 1
            except Exception:
                nctr_s = 1
            try:
                nctr_p = (len(mol.c2[p0:p1]) // nprim) if hasattr(mol, 'c2') and mol.c2 is not None else 0
            except Exception:
                nctr_p = 0
            nfunc = int(nctr_s * 1 + nctr_p * 3)
        else:
            # regular shell: number of contraction columns inferred from c1 slice length
            try:
                nctr = (len(mol.c1[p0:p1]) // nprim) if hasattr(mol, 'c1') and mol.c1 is not None else 1
            except Exception:
                nctr = 1
            degen = mult_map.get(st, mult_map.get(abs(st), 0))
            nfunc = int(nctr * degen)

        ao_loc_shell.append(ao_loc_shell[-1] + nfunc)
    ao_loc_shell = _np.array(ao_loc_shell, dtype=int)

    # Now group per atom and per L (treat SP (-1) as two groups: L=0 (first AO) and L=1 (remaining))
    shell_idx_list = []
    for ia in range(natm):
        # shells on this atom in FCHK order
        shells = [ish for ish in range(nbas) if shell_atom[ish] == ia]
        if not shells:
            continue
        # collect L values present (use absolute value except keep -1 to split)
        Lset = set()
        for ish in shells:
            st = shell_types[ish]
            if st == -1:
                Lset.update([0, 1])
            else:
                Lset.add(abs(st))
        for L in sorted(Lset):
            idx = []
            for ish in shells:
                st = shell_types[ish]
                start = int(ao_loc_shell[ish])
                end = int(ao_loc_shell[ish + 1])
                if st == -1:
                    # SP: S is first AO, P are remaining
                    if L == 0:
                        idx.extend(range(start, start + 1))
                    elif L == 1:
                        idx.extend(range(start + 1, end))
                else:
                    if abs(st) == L:
                        idx.extend(range(start, end))
            shell_idx_list.append(_np.array(idx, dtype=int))

    return shell_idx_list

def aoslice_by_atom_from_bas(m):
    """Return list of (iat, dummy, ao_start, ao_end) following m._bas order."""
    bas = np.asarray(m._bas)
    # ao_loc in the same order as _bas (start index per shell)
    if hasattr(m, 'ao_loc'):
        ao_loc = np.asarray(m.ao_loc)
    else:
        ao_loc = np.asarray(m.ao_loc_nr())

    # group shells by atom (assumes bas[:,0] uses 0-based atom indices)
    shells_by_atom = {}
    for ish, brow in enumerate(bas):
        iat = int(brow[0])
        shells_by_atom.setdefault(iat, []).append(ish)

    slices = []
    natm = int(getattr(m, 'natm', getattr(m, 'natoms', None)))
    for iat in range(natm):
        shells = shells_by_atom.get(iat, [])
        if not shells:
            slices.append((iat, 0, 0, 0))
            continue
        p0 = int(ao_loc[shells[0]])
        p1 = int(ao_loc[shells[-1] + 1])
        slices.append((iat, 0, p0, p1))
    return slices


import numpy as np
from functools import reduce

import numpy as np

def permute_aos(mat, mol):
    cart_flag = getattr(mol, 'cart', False)

    # choose the _bas that corresponds to the matrix:
    # if this is MINAO object use its _bas, else use underlying mol._bas
    bas = mol._bas

    final_order = []
    offset = 0

    for basrow in bas:
        l = int(basrow[1])
        nctr = int(basrow[3])   # number of contractions for this shell

        if cart_flag:
            per_ctr = (l + 1) * (l + 2) // 2
            key = l
        else:
            per_ctr = 2 * l + 1
            key = -l

        mapping = MAPS.get(key)

        # total functions contributed by this shell:
        total_funcs = per_ctr * nctr

        if mapping is None:
            # identity: include all functions for all contractions
            final_order.extend(range(offset, offset + total_funcs))
        else:
            # mapping describes reorder for ONE contraction block of length per_ctr.
            # apply the mapping for each contraction block, shifted by per_ctr*k
            for k in range(nctr):
                block_offset = offset + k * per_ctr
                # invert mapping: mapping[old] = new; we need inv[new] = old
                inv = [0] * per_ctr
                for old, new in enumerate(mapping):
                    inv[new] = old
                for idx in inv:
                    final_order.append(block_offset + idx)

        offset += total_funcs

    if len(final_order) != mat.shape[0]:
        raise ValueError(f"permute_aos: final_order length {len(final_order)} != matrix dim {mat.shape[0]}. "
                         "Use the same _bas that was used to build the matrix.")

    return mat[np.ix_(final_order, final_order)]

def permute_aos_cross(mat, mol1, mol2):
    """Permute rows/columns of cross-overlap matrix for mf1 and mf2 independently."""
    def make_order(mol):
        cart_flag = getattr(mol, 'cart', False)
        bas = mol._bas if getattr(mol, "minao", False) else mol.mol._bas
        final_order = []
        offset = 0
        for basrow in bas:
            l = int(basrow[1])
            nctr = int(basrow[3])
            per_ctr = (l + 1) * (l + 2) // 2 if cart_flag else 2 * l + 1
            key = l if cart_flag else -l
            mapping = MAPS.get(key)
            total_funcs = per_ctr * nctr
            if mapping is None:
                final_order.extend(range(offset, offset + total_funcs))
            else:
                for k in range(nctr):
                    block_offset = offset + k * per_ctr
                    inv = [0] * per_ctr
                    for old, new in enumerate(mapping):
                        inv[new] = old
                    for idx in inv:
                        final_order.append(block_offset + idx)
            offset += total_funcs
        return final_order

    row_order = make_order(mol1)
    col_order = make_order(mol2)
    return mat[np.ix_(row_order, col_order)]

def permute_aos_cols(pre_orth_ao, mol):
    """
    Return a copy of pre_orth_ao where, for each shell in mol._bas (or mol.mol._bas
    if using MoleANO/minao), the functions belonging to that shell have their
    *columns* permuted according to MAPS (applied per contraction block).
    Rows are left untouched.
    """
    bas = mol._bas
    cart_flag = getattr(mol, 'cart', False)

    col_order = []
    offset = 0

    for basrow in bas:
        l = int(basrow[1])
        nctr = int(basrow[3])   # number of contractions for this shell

        if cart_flag:
            per_ctr = (l + 1) * (l + 2) // 2
            key = l
        else:
            per_ctr = 2 * l + 1
            key = -l

        mapping = MAPS.get(key)
        total_funcs = per_ctr * nctr

        if mapping is None:
            col_order.extend(range(offset, offset + total_funcs))
        else:
            inv = [0] * per_ctr
            for old, new in enumerate(mapping):
                inv[new] = old
            for k in range(nctr):
                block_offset = offset + k * per_ctr
                for idx in inv:
                    col_order.append(block_offset + idx)

        offset += total_funcs

    # Apply permutation to columns ONLY:
    new = pre_orth_ao[:, col_order]
    return new

import numpy as np

def permute_aos_rows(mat, mole2):
    """
    Permute FCHK AO/MO columns into PySCF AO order.

    Requirements on mole2:
      - mole2.pyscf_mol  : pyscf.gto.Mole instance (built)
      - mole2.fchk_basis_arrays : dict with keys 'iatsh' (1-based) and 'mssh'
      - mole2.coord or mole2.fchk_coords : Nx3 array (FCHK atom coords, same units as pyscf.mol.atom?)
      - mole2.cart : boolean (True if FCHK was written with cartesian high-L shells)
    """

    mol = mole2.pyscf_mol
    cart_pyscf = bool(getattr(mol, 'cart', False))
    cart_fchk  = bool(getattr(mole2, 'cart', False))

    # --- per-shell local reorder maps (Gaussian -> PySCF)
    MAPS = {
        # spherical keys (negative l)
        -2: [0, 1, 2, 3, 4],
        -3: [0, 2, 1, 4, 3, 5, 6],
        -4: [0, 2, 1, 4, 3, 6, 5, 8, 7],
        # cartesian keys (positive l)
        2: [0, 3, 5, 1, 4, 2],
        3: [0, 5, 7, 1, 3, 8, 4, 6, 2, 9],
        4: [0, 5, 9, 1, 6, 10, 2, 7, 11, 3, 8, 12, 4, 13, 14],
    }

    # --- Build FCHK gaussian shell list: list of (atom0, l, start_index, nfuncs) ---
    iatsh = mole2.fchk_basis_arrays['iatsh']  # 1-based
    mssh  = mole2.fchk_basis_arrays['mssh']
    g_shells = []
    ptr = 0
    for iat, mst in zip(iatsh, mssh):
        atom0 = int(iat) - 1
        mst = int(mst)
        if mst == -1:
            entries = [(0,1),(1,3)]
        elif mst == 0:
            entries = [(0,1)]
        elif mst == 1:
            entries = [(1,3)]
        else:
            l = abs(mst)
            if mst < 0 or not cart_fchk:
                n = 2*l + 1
            else:
                n = (l + 1) * (l + 2) // 2
            entries = [(l, n)]
        for l, n in entries:
            g_shells.append((atom0, l, ptr, int(n)))
            ptr += int(n)
    total_fchk_funcs = ptr

    # --- Map FCHK atom indices -> PySCF atom indices by matching coordinates ---
    # Use coordinates to map (robust vs differing atom orderings). Expect mole2.coord exists.
    if hasattr(mole2, 'coord'):
        fcoords = np.asarray(mole2.coord)
    elif hasattr(mole2, 'fchk_coords'):
        fcoords = np.asarray(mole2.fchk_coords)
    else:
        # fallback: assume atom numbering matches
        fcoords = None

    if fcoords is not None:
        pyscf_coords = np.asarray(mol.atom_coords())
        # compute distance matrix and greedily match nearest neighbors
        dmat = np.linalg.norm(fcoords[:,None,:] - pyscf_coords[None,:,:], axis=2)
        # Hungarian matching ensures one-to-one
        try:
            from scipy.optimize import linear_sum_assignment
            row, col = linear_sum_assignment(dmat)
            fchk_to_pyscf = {int(r): int(c) for r,c in zip(row, col)}
        except Exception:
            # greedy fallback
            fchk_to_pyscf = {}
            used = set()
            for i in range(dmat.shape[0]):
                j = int(np.argmin(dmat[i]))
                # if collision, find next best
                if j in used:
                    order = np.argsort(dmat[i])
                    for cand in order:
                        if cand not in used:
                            j = int(cand); break
                used.add(j)
                fchk_to_pyscf[i] = j
    else:
        # assume direct mapping
        fchk_to_pyscf = {i:i for i in range(max(1, mol.natm))}

    # Now translate g_shells atom index from FCHK atom to PySCF atom index
    g_shells_mapped = [(fchk_to_pyscf.get(atom, atom), l, start, n) for (atom,l,start,n) in g_shells]

    # Build an index: for each (atom,l) list of starts in gaussian order
    g_reg = {}
    for atom, l, start, n in g_shells_mapped:
        g_reg.setdefault((atom, l), []).append((start, n))

    # --- Walk PySCF shells in their order and consume from g_reg ---
    perm = []
    consumed = {}
    for ib in range(mol.nbas):
        atom_idx = int(mol.bas_atom(ib))
        l_val = int(mol.bas_angular(ib))

        # pyscf number of components for this shell
        if l_val == 0:
            n_pyscf = 1
        elif l_val == 1:
            n_pyscf = 3
        else:
            n_pyscf = (l_val + 1)*(l_val + 2)//2 if cart_pyscf else 2*l_val + 1

        key = (atom_idx, l_val)
        consumed.setdefault(key, 0)
        c = consumed[key]

        if key not in g_reg or c >= len(g_reg[key]):
            raise ValueError(f"Missing Gaussian shell for atom={atom_idx}, L={l_val}: have {len(g_reg.get(key,[]))}, needed #{c+1}")

        start, n_fchk = g_reg[key][c]
        consumed[key] += 1

        # verify counts (typical error: spherical<->cart mismatch)
        if n_fchk != n_pyscf:
            raise ValueError(
                f"Mismatch component counts for atom {atom_idx} L={l_val}: "
                f"FCHK has {n_fchk}, PySCF expects {n_pyscf} (mol.cart={cart_pyscf}, fchk.cart={cart_fchk})."
            )

        # choose local map: spherical -> MAPS[-l], cart -> MAPS[l]
        map_key = -l_val if not cart_pyscf else l_val
        local_map = MAPS.get(map_key, None)
        if local_map is None:
            local_map = list(range(n_pyscf))

        if len(local_map) != n_pyscf:
            raise RuntimeError(f"Local MAP length mismatch for L={l_val}: {len(local_map)} != {n_pyscf}")

        for idx in local_map:
            perm.append(start + idx)

    # final sanity
    if len(perm) != total_fchk_funcs:
        raise RuntimeError(f"Permutation length {len(perm)} != total FCHK functions {total_fchk_funcs}")

    perm = np.asarray(perm, int)

    # Apply permutation to mat (supports typical shapes)
    if mat.ndim == 2 and mat.shape[1] == len(perm):
        return mat[:, perm]
    if mat.ndim == 2 and mat.shape[0] == len(perm):
        return mat[perm, :]
    if mat.ndim == 3 and mat.shape[2] == len(perm):
        return mat[:, :, perm]

    raise ValueError("Matrix shape incompatible with AO permutation length.")

