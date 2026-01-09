from os import environ
import numpy as np
from collections import deque, defaultdict

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

    # Include connectivity if a value was passed (allow empty dicts)
    if connec is not None:
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
    import numpy as _np
    print(" | Obtaining Natural Orbitals from the 1-RDM...")

    # Ask the mean-field object for the 1-RDM in AO representation
    rdm1 = mf.make_rdm1(ao_repr=True)

    # Convert to numpy and handle a few possible shapes/types
    rdm1_arr = _np.asarray(rdm1)

    if rdm1_arr.ndim == 3:
        #raise NotImplementedError(" | Natural Orbitals for unrestricted calculations are not implemented yet.")
        D = _np.sum(rdm1_arr, axis=0)
    elif rdm1_arr.ndim == 2:
        D = rdm1_arr
    else:
        raise ValueError(" | Could not find dimensions for the 1-RDM")

    # Symmetrize S as well and coerce to float
    S = _np.asarray(S, dtype=float)
    S = 0.5 * (S + S.T)

    # Solve generalized eigenproblem to obtain NOs and occupancies
    occ, coeff = eigh(_np.linalg.multi_dot((S, D, S)), b=S)

    # Order descending
    coeff = coeff[:, ::-1]
    occ = occ[::-1]

    # Zero out tiny negative occupancies
    occ[occ < 1e-12] = 0.0

    # Return occupations as diagonal matrix (keeps previous API)
    occ_diag = _np.diag(occ)
    return occ_diag, coeff


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
    Reorders FCHK AO rows to match PySCF's internal layout AND applies
    Mokit-style normalization scaling for Cartesian basis sets.
    """
    mol = mole2.pyscf_mol
    is_cart = bool(getattr(mol, 'cart', False))

    # --- 1. Define Mokit Scaling Constants (Sdiag) ---
    # These map Gaussian Cartesian norms to PySCF Cartesian norms
    PI = np.pi
    p1 = 2.0 * np.sqrt(PI / 15.0)
    p2 = 2.0 * np.sqrt(PI / 5.0)
    p3 = 2.0 * np.sqrt(PI / 7.0)
    p4 = 2.0 * np.sqrt(PI / 35.0)
    p5 = 2.0 * np.sqrt(PI / 105.0)
    p6 = (2.0 / 3.0) * np.sqrt(PI)
    p7 = (2.0 / 3.0) * np.sqrt(PI / 7.0)
    p8 = (2.0 / 3.0) * np.sqrt(PI / 35.0)
    p9 = 2.0 * np.sqrt(PI / 11.0)
    p10 = (2.0 / 3.0) * np.sqrt(PI / 11.0)
    p11 = 2.0 * np.sqrt(PI / 231.0)
    p12 = (2.0 / 3.0) * np.sqrt(PI / 77.0)
    p13 = 2.0 * np.sqrt(PI / 1155.0)

    # Scaling arrays ordered according to PySCF Internal Order
    # Mokit logic: coeff_pyscf = coeff_fchk / Sdiag
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

    # --- 2. Define Permutation Maps ---
    MAPS = {
        # Cartesian (Gaussian -> PySCF)
        2: [0, 3, 4, 1, 5, 2],
        3: [0, 4, 5, 3, 9, 6, 1, 8, 7, 2],
        4: [0, 4, 5, 3, 14, 6, 11, 13, 12, 9, 1, 8, 7, 10, 2],
        5: [0, 5, 6, 4, 20, 7, 15, 19, 18, 11, 10, 14, 13, 17, 16, 1, 9, 8, 12, 11, 2],
        6: [0, 6, 7, 5, 27, 8, 19, 26, 25, 13, 15, 22, 21, 24, 23, 10, 18, 17, 20, 19, 16, 1, 10, 9, 14, 13, 11, 2],

        # Spherical (Gaussian -> PySCF)
        -2: [4, 2, 0, 1, 3],
        -3: [6, 4, 2, 0, 1, 3, 5],
        -4: [8, 6, 4, 2, 0, 1, 3, 5, 7],
        -5: [10, 8, 6, 4, 2, 0, 1, 3, 5, 7, 9],
        -6: [12, 10, 8, 6, 4, 2, 0, 1, 3, 5, 7, 9, 11]
    }

    # --- 3. Build Permutation and Scaling Lists ---
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

    # --- 4. Apply to Matrix ---
    p = np.array(idx_list)
    s = np.array(scale_list)

    # mat shape is typically (NAO, NMO) or (NMO, NAO) or (Spin, ...).
    # We detect dimension matching `len(p)` to find the AO axis.

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
