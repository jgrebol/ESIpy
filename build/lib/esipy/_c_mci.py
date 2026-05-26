import numpy as np
import warnings

try:
    from numba import njit, prange
    _HAS_NUMBA = True
except ImportError:
    warnings.warn("Numba not found. Install it for performance: pip install numba")
    _HAS_NUMBA = False


def prepare_matrices(aom_list, occ=None):
    aoms = [a.astype(np.float64) for a in aom_list]
    if occ is not None:
        occ = np.asarray(occ, dtype=np.float64)
        aoms = [np.dot(occ, a) for a in aoms]
    return np.array(aoms, dtype=np.float64, order='C')


if _HAS_NUMBA:
    @njit(fastmath=True)
    def _trace_dot(A, B, m):
        val = 0.0
        for i in range(m):
            s = 0.0
            for k in range(m):
                s += A[i, k] * B[k, i]
            val += s
        return val

    @njit(parallel=True, fastmath=True)
    def mci_dfs_kernel(matrices):
        n_atoms = matrices.shape[0]
        m_size = matrices.shape[1]
        total_mci = 0.0
        mat0 = matrices[0]

        for second_atom in prange(1, n_atoms):
            mat_stack = np.zeros((n_atoms - 1, m_size, m_size), dtype=np.float64)
            mat_stack[1][:, :] = np.dot(mat0, matrices[second_atom])

            path = np.zeros(n_atoms, dtype=np.int64)
            path[0] = 0
            path[1] = second_atom
            stack_idx = np.zeros(n_atoms, dtype=np.int64)
            stack_idx[2] = 1
            current_mask = (1 << 0) | (1 << second_atom)
            depth = 2
            local_sum = 0.0

            while depth >= 2:
                candidate = stack_idx[depth]
                if candidate >= n_atoms:
                    depth -= 1
                    if depth >= 2:
                        prev_atom = path[depth]
                        current_mask &= ~(1 << prev_atom)
                        stack_idx[depth] += 1
                    continue

                if (current_mask >> candidate) & 1:
                    stack_idx[depth] += 1
                    continue

                if depth == n_atoms - 1:
                    if second_atom < candidate:
                        local_sum += _trace_dot(mat_stack[depth - 1], matrices[candidate], m_size)
                    stack_idx[depth] += 1
                else:
                    path[depth] = candidate
                    current_mask |= (1 << candidate)
                    mat_stack[depth][:, :] = np.dot(mat_stack[depth - 1], matrices[candidate])
                    depth += 1
                    stack_idx[depth] = 1

            total_mci += local_sum

        return total_mci


    def compute_mci_numba(arr, aom, occ=None):
        if not _HAS_NUMBA:
            raise RuntimeError("Numba is required for this function.")

        if isinstance(aom, (tuple, list)) and len(aom) == 2 and hasattr(aom[1], 'shape'):
            real_aoms = aom[0]
            occupation = aom[1]
        else:
            real_aoms = aom
            occupation = occ

        # small guard: trivial rings
        if len(arr) < 2:
            return 0.0

        # convert to numpy array for faster indexing if possible
        try:
            real_aoms_arr = np.asarray(real_aoms)
            indices = (np.asarray(arr, dtype=np.int64) - 1)
            ring_matrices = real_aoms_arr[indices]
        except Exception:
            ring_matrices = [real_aoms[i - 1] for i in arr]

        flat_matrices = prepare_matrices(ring_matrices, occupation)
        return mci_dfs_kernel(flat_matrices)


    def compute_iring_numba(arr, aom, occ=None):
        if isinstance(aom, (tuple, list)) and len(aom) == 2 and hasattr(aom[1], 'shape'):
            real_aoms = aom[0]
            occupation = aom[1]
        else:
            real_aoms = aom
            occupation = occ

        if len(arr) < 1:
            return 0.0

        try:
            real_aoms_arr = np.asarray(real_aoms)
            indices = (np.asarray(arr, dtype=np.int64) - 1)
            ring_matrices = real_aoms_arr[indices]
        except Exception:
            ring_matrices = [real_aoms[i - 1] for i in arr]

        if occupation is not None:
            ring_matrices = [np.dot(occupation, m) for m in ring_matrices]

        prod = np.eye(ring_matrices[0].shape[0])
        for mtx in ring_matrices:
            prod = np.dot(prod, mtx)

        return (2 ** (len(arr) - 1)) * np.trace(prod)


else:
    from itertools import permutations

    def _trace_of_permutation_product_py(aoms, perm):
        P = aoms[perm[0]]
        for i in range(1, len(perm)):
            P = P.dot(aoms[perm[i]])
        return float(np.trace(P))

    def _compute_total_py(aoms):
        total = 0.0
        for perm in permutations(range(aoms.shape[0])):
            total += _trace_of_permutation_product_py(aoms, perm)
        return total

    def compute_mci_numba(arr, aom, occ=None):
        if isinstance(aom, (tuple, list)) and len(aom) == 2 and hasattr(aom[1], 'shape'):
            real_aoms = aom[0]
            occupation = aom[1]
        else:
            real_aoms = aom
            occupation = occ

        if len(arr) < 2:
            return 0.0

        try:
            real_aoms_arr = np.asarray(real_aoms)
            indices = (np.asarray(arr, dtype=np.int64) - 1)
            ring_matrices = real_aoms_arr[indices]
        except Exception:
            ring_matrices = [real_aoms[i - 1] for i in arr]

        flat_matrices = prepare_matrices(ring_matrices, occupation)
        return _compute_total_py(flat_matrices)

    def compute_iring_numba(arr, aom, occ=None):
        if isinstance(aom, (tuple, list)) and len(aom) == 2 and hasattr(aom[1], 'shape'):
            real_aoms = aom[0]
            occupation = aom[1]
        else:
            real_aoms = aom
            occupation = occ

        if len(arr) < 1:
            return 0.0

        try:
            real_aoms_arr = np.asarray(real_aoms)
            indices = (np.asarray(arr, dtype=np.int64) - 1)
            ring_matrices = real_aoms_arr[indices]
        except Exception:
            ring_matrices = [real_aoms[i - 1] for i in arr]

        if occupation is not None:
            ring_matrices = [np.dot(occupation, m) for m in ring_matrices]

        prod = np.eye(ring_matrices[0].shape[0])
        for mtx in ring_matrices:
            prod = np.dot(prod, mtx)

        return (2 ** (len(arr) - 1)) * np.trace(prod)
