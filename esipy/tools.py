from collections import deque, defaultdict
import numpy as np

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

    # Include connectivity if a value was passed
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
    elif partition in ["iao-autosad-freeatom", "autosad-freeatom", "iaofree", "iaof", "iaf", "f", "autosad", "iaosad", "if", "iaoaf", "autos"]:
        return "iao-autosad-freeatom"
    elif partition in ["iao-autosad", "iao-autosad-mull", "autosad-mull", "iaomull", "iaom", "iam", "iaomul", "autosadmull",
                           "iaosad", "im", "autos"]: # Make it the default AUTOSAD procedure
        return "iao-autosad-mull"
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

    # Order descending
    coeff = coeff[:, ::-1]
    occ = occ[::-1]
    occ[occ < 1e-12] = 0.0 # Small values set to zero
    occ_diag = np.diag(occ)
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


def iao(mol, pmol, coeffs):
    """
    Build IAOs using Knizia's exact depolarization formula.
    Numerically stable to guarantee exact density conservation.
    """
    from pyscf.gto.mole import intor_cross
    from pyscf.lo import orth
    import scipy.linalg
    import numpy as np

    # 1. Overlap Matrices
    S1 = mol.intor('int1e_ovlp')
    S2 = pmol.intor('int1e_ovlp')
    S12 = intor_cross('int1e_ovlp', mol, pmol)

    # Number of occupied orbitals
    nocc = mol.nelectron // 2
    C_occ = coeffs[:, :nocc]  # (nbas x nocc)

    # 2. Represent minimal basis in full AO basis: A_tilde = S1^-1 * S12
    # Using scipy.linalg.solve for numerical stability
    A_tilde = scipy.linalg.solve(S1, S12, assume_a='pos')

    # 3. Project occupied MOs onto minimal basis
    # C_min = S2^-1 * S21 * C_occ
    S21_Cocc = np.dot(S12.T, C_occ)
    C_min = scipy.linalg.solve(S2, S21_Cocc, assume_a='pos')

    # Express projected occupied MOs back in AO basis
    C_tilde = np.dot(A_tilde, C_min)

    # Orthogonalize the projected occupied MOs
    C_proj = orth.vec_lowdin(C_tilde, S1)

    # 4. Form IAOs (Knizia Eq. 10 optimized for stability)
    # Instead of full projector matrices (nbas x nbas), we apply them directly to A_tilde

    # P_occ @ A_tilde = C_occ @ (C_occ.T @ S12)
    P_occ_A = np.dot(C_occ, np.dot(C_occ.T, S12))

    # P_proj @ A_tilde = C_proj @ (C_proj.T @ S12)
    P_proj_A = np.dot(C_proj, np.dot(C_proj.T, S12))

    # P_occ @ P_proj @ A_tilde
    P_occ_P_proj_A = np.dot(C_occ, np.dot(C_occ.T, np.dot(S1, P_proj_A)))

    # IAO = A_tilde + 2*(P1 P2 A) - P1 A - P2 A
    IAO_nonorth = A_tilde + 2 * P_occ_P_proj_A - P_occ_A - P_proj_A

    # 5. Final Symmetric Orthogonalization
    IAOs = orth.vec_lowdin(IAO_nonorth, S1)

    return IAOs

def get_effaos(mol, mf, coeffs, free_atom=True):
    """
    Builds effective Atomic Orbitals (eff-AOs) and their occupations.
    Modified to handle Unrestricted (UKS/UHF) alpha and beta orbitals.
    """
    from pyscf.lo.nao import _prenao_sub
    from pyscf.data import elements
    import numpy as np
    from pyscf import gto, scf

    # Number of unpaired electrons for ground state atoms
    atom_spins = {
        'H': 1, 'He': 0, 'Li': 1, 'Be': 0, 'B': 1, 'C': 2, 'N': 3, 'O': 2, 'F': 1, 'Ne': 0,
        'Na': 1, 'Mg': 0, 'Al': 1, 'Si': 2, 'P': 3, 'S': 2, 'Cl': 1, 'Ar': 0,
        'K': 1, 'Ca': 0, 'Sc': 1, 'Ti': 2, 'V': 3, 'Cr': 6, 'Mn': 5, 'Fe': 4, 'Co': 3, 'Ni': 2, 'Cu': 1, 'Zn': 0,
        'Ga': 1, 'Ge': 2, 'As': 3, 'Se': 2, 'Br': 1, 'Kr': 0
    }

    # Standard Minimal Basis sizes
    def get_num_minbas(sym):
        z = elements.charge(sym)
        if z <= 2: return 1  # H-He
        if z <= 10: return 5  # Li-Ne
        if z <= 18: return 9  # Na-Ar
        if z <= 30: return 18  # K-Zn
        raise NotImplementedError(f"Minimal basis size not defined for element: {sym}")

    aoslices = mol.aoslice_by_atom()
    atom_syms = [mol.atom_pure_symbol(i) for i in range(mol.natm)]

    # Calculate total minimal basis size
    minbas_sizes = []
    for sym in atom_syms:
        n_min = get_num_minbas(sym)
        minbas_sizes.append(n_min)

    # Initialize outputs for potentially 2 spins
    total_min_dim = sum(minbas_sizes)
    veps_block = np.zeros((mol.nao, total_min_dim))
    vaps_diag = []
    S_mol = mol.intor("int1e_ovlp")
    if "U" in mf.__class__.__name__:
        dm = mf.make_rdm1(ao_repr=True)
        P_mol = dm[0] + dm[1] if dm.ndim == 3 else dm
    else:
        P_mol = mf.make_rdm1(ao_repr=True)

    # Loop over spin channels (alpha=0, beta=1)
    col_idx = 0
    atom_dict = {}
    for ia, sym in enumerate(atom_syms):
        p0, p1 = aoslices[ia, 2], aoslices[ia, 3]
        n_target = minbas_sizes[ia]

        # Create Mole with only this atom
        mol_atm = gto.Mole()
        mol_atm.atom = f"{sym} 0.0 0.0 0.0"
        mol_atm.basis = {sym: mol._basis[sym]}
        mol_atm.spin = atom_spins.get(sym, 0)
        mol_atm.verbose = 0
        mol_atm.cart = mol.cart
        mol_atm.build()


        if free_atom:
            if sym not in atom_dict:
                # Inherit DFT/HF method, upgrading to Unrestricted if needed
                if "dft" in mf.__module__:
                    mf_atm = scf.KS(mol_atm)
                    if hasattr(mf, 'xc'): mf_atm.xc = mf.xc
                else:
                    mf_atm = scf.HF(mol_atm)

                # Compute the SCF for the free atom
                mf_atm.kernel()

                # Get Density/Overlap
                dm_local = mf_atm.make_rdm1()
                P_local = dm_local[0] + dm_local[1] if dm_local.ndim == 3 else dm_local
                S_local = mf_atm.get_ovlp()

                # Calculate
                PS = np.dot(P_local, S_local)
                SPS = np.dot(S_local, PS)
                w, c = _prenao_sub(mol_atm, SPS, S_local)

                # Sort and Truncate
                idx = np.argsort(w)[::-1][:n_target]
                c_sel = c[:, idx]
                atom_dict[sym] = (w[idx], c_sel)

            w_keep, c_keep = atom_dict[sym]

        else:
            # Trossets de la P i S per a aquest spin
            P_local = P_mol[p0:p1, p0:p1]
            S_local = S_mol[p0:p1, p0:p1]

            PS = np.dot(P_local, S_local)
            SPS = np.dot(S_local, PS)

            # Spherical average dels trossets de la PS
            w, c = _prenao_sub(mol_atm, SPS, S_local)

            # Sort descending by occupation
            idx = np.argsort(w)[::-1][:n_target]
            w_keep = w[idx]
            c_keep = c[:, idx]

        # Fiquem el bloc dels coeficients al trosset que toca
        veps_block[p0:p1, col_idx: col_idx + n_target] = c_keep
        vaps_diag.extend(w_keep)

        col_idx += n_target

    # Return simplified shapes if restricted
    return np.array(vaps_diag), veps_block

def autosad(mol, mf, free_atom=True):
    """
    Builds IAO-AutoSAD orbitals.
    Handles both Restricted (returns array) and Unrestricted (returns tuple of arrays).
    """
    from scipy.linalg import solve
    from pyscf.lo.orth import vec_lowdin
    import numpy as np

    def do_autosad(mol, mf, C_occ, free_atom):
        S1 = mol.intor('int1e_ovlp')
        aaaa, effaos = get_effaos(mol, mf, coeffs=C_occ, free_atom=free_atom)

        # Effective overlaps
        S12_eff = np.dot(S1, effaos)
        S2_eff = np.linalg.multi_dot((effaos.T, S1, effaos))

        A_tilde = effaos

        S21_Cocc = np.dot(S12_eff.T, C_occ)
        C_min = solve(S2_eff, S21_Cocc, assume_a='pos')
        C_proj = vec_lowdin(np.dot(A_tilde, C_min), S1)

        # Form IAOs (Knizia Eq 10)
        P_occ_A = np.dot(C_occ, np.dot(C_occ.T, S12_eff))
        P_proj_A = np.dot(C_proj, np.dot(C_proj.T, S12_eff))
        P_occ_P_proj_A = np.dot(C_occ, np.dot(C_occ.T, np.dot(S1, P_proj_A)))

        IAO_nonorth = A_tilde + 2 * P_occ_P_proj_A - P_occ_A - P_proj_A
        return vec_lowdin(IAO_nonorth, S1)

    if "U" in mf.__class__.__name__:
        coeff_alpha, coeff_beta = mf.mo_coeff
        autosad_alpha = do_autosad(mol, mf, coeff_alpha[:, :mf.nelec[0]], free_atom)
        autosad_beta = do_autosad(mol, mf, coeff_beta[:, :mf.nelec[1]], free_atom)
        return autosad_alpha, autosad_beta
    else:
        coeffs = mf.mo_coeff
        autosad = do_autosad(mol, mf, coeffs[:, :mol.nelectron // 2], free_atom)
        return autosad


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
