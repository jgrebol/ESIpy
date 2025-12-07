"""
Simplified _c_mci.py: unified, minimal wrappers for the C MCI routines.
The module attempts to load a Python extension `esipy.mci` first; if not found,
it looks for a shared object file in the package directory and wraps it via
`ctypes.CDLL`.

The C API expected (ctypes) is:
  double compute_mci_sym(const int *ring, int ring_len, const double *aoms_stacked, int naoms, int m)
  double compute_mci_natorbs_sym(const int *ring, int ring_len, const double *aoms_stacked, int naoms, int m, const double *occ)
  double compute_mci_nosym(const int *ring, int ring_len, const double *aoms_stacked, int naoms, int m)
  double compute_mci_natorbs_nosym(const int *ring, int ring_len, const double *aoms_stacked, int naoms, int m, const double *occ)

This file stacks the AOMs (Fortran-order) and then delegates to the C routines.
"""
import os
import ctypes
from ctypes import c_int, c_double, POINTER
import numpy as np

_lib = None
_LIB_PATH = None

# Try to import compiled extension module first (python extension providing functions)
try:
    import esipy.mci as libmci_ext  # type: ignore
    _lib = libmci_ext
    _LIB_PATH = getattr(libmci_ext, '__file__', None)
except Exception:
    _lib = None
    _LIB_PATH = None

# If not found, try to locate a shared library in the package dir
if _lib is None:
    here = os.path.dirname(__file__)
    candidates = [os.path.join(here, 'libmci.so'), os.path.join(here, 'libmci.dylib'), os.path.join(here, 'libc_mci.so')]
    for p in candidates:
        if os.path.exists(p):
            try:
                _lib = ctypes.CDLL(p)
                _LIB_PATH = p
                break
            except Exception:
                _lib = None

# If we loaded a ctypes library, set prototypes
_is_ctypes = isinstance(_lib, ctypes.CDLL)
if _is_ctypes and _lib is not None:
    try:
        _lib.compute_mci_sym.restype = c_double
        _lib.compute_mci_sym.argtypes = [POINTER(c_int), c_int, POINTER(c_double), c_int, c_int]

        _lib.compute_mci_natorbs_sym.restype = c_double
        _lib.compute_mci_natorbs_sym.argtypes = [POINTER(c_int), c_int, POINTER(c_double), c_int, c_int, POINTER(c_double)]

        _lib.compute_mci_nosym.restype = c_double
        _lib.compute_mci_nosym.argtypes = [POINTER(c_int), c_int, POINTER(c_double), c_int, c_int]

        _lib.compute_mci_natorbs_nosym.restype = c_double
        _lib.compute_mci_natorbs_nosym.argtypes = [POINTER(c_int), c_int, POINTER(c_double), c_int, c_int, POINTER(c_double)]
    except Exception:
        # If prototypes fail, treat as unavailable
        _lib = None
        _is_ctypes = False


def has_c_module():
    """Return True if a C-backed MCI implementation is available."""
    return _lib is not None


# ----------------- helpers -----------------

def _stack_aoms(aom_list):
    """Stack list of (m x m) AOMs into a 1-D Fortran-ordered float64 array.
    The C side expects stacked array of length na * m * m in column-major (Fortran) order.
    """
    if len(aom_list) == 0:
        raise ValueError('aom_list must be non-empty')
    # ensure we capture the shape from the first AOM (avoid complex inline expressions for type checkers)
    first = np.asarray(aom_list[0])
    m = int(first.shape[0])
    na = len(aom_list)
    stacked = np.empty((na * m * m,), dtype=np.float64, order='F')
    for i, A in enumerate(aom_list):
        Af = np.asfortranarray(np.asarray(A, dtype=np.float64))
        if Af.shape != (m, m):
            raise ValueError('All AOMs must be square with same shape')
        # place in F-order block for atom i
        stacked[i * m * m: (i + 1) * m * m] = Af.ravel(order='F')
    return stacked


def _make_ring(n):
    """Return a ctypes array (c_int * n) filled with 0..n-1 (C API expects positions in ring order)."""
    return (c_int * n)(*range(n))


# ----------------- unified call wrappers -----------------

def _call_ctypes_sym(ring_ctypes, n, stacked_ptr, na, m):
    return float(_lib.compute_mci_sym(ring_ctypes, c_int(n), stacked_ptr, c_int(na), c_int(m)))


def _call_ctypes_natorb_sym(ring_ctypes, n, stacked_ptr, na, m, occ_ptr):
    return float(_lib.compute_mci_natorbs_sym(ring_ctypes, c_int(n), stacked_ptr, c_int(na), c_int(m), occ_ptr))


def _call_ctypes_nosym(ring_ctypes, n, stacked_ptr, na, m):
    return float(_lib.compute_mci_nosym(ring_ctypes, c_int(n), stacked_ptr, c_int(na), c_int(m)))


def _call_ctypes_natorb_nosym(ring_ctypes, n, stacked_ptr, na, m, occ_ptr):
    return float(_lib.compute_mci_natorbs_nosym(ring_ctypes, c_int(n), stacked_ptr, c_int(na), c_int(m), occ_ptr))


# If the loaded lib is a Python extension module (not ctypes), we'll call functions directly.
# We expect the extension to accept (ring_array, n, stacked, na, m[, occ]) where `stacked` can be
# a numpy Fortran-ordered array.


# ----------------- public APIs -----------------

def compute_mci_sym(aom_for_ring):
    """Compute MCI for symmetric AOMs (regular Iring) using the C backend.
    `aom_for_ring` is a list/sequence of m×m arrays.
    """
    if _lib is None:
        raise RuntimeError('C MCI module not available')
    n = len(aom_for_ring)
    if n < 2:
        return 0.0
    m = int(np.asarray(aom_for_ring[0]).shape[0])
    stacked = _stack_aoms(aom_for_ring)

    if _is_ctypes:
        ring = _make_ring(n)
        stacked_ptr = stacked.ctypes.data_as(POINTER(c_double))
        return _call_ctypes_sym(ring, n, stacked_ptr, n, m)
    else:
        # python extension: call function with numpy arrays
        return float(_lib.compute_mci_sym(np.arange(n, dtype=np.int32), n, stacked, n, m))


def compute_mci_nosym(aom_for_ring):
    """Compute MCI for non-symmetric AOMs (regular Iring with reverse-filter)."""
    if _lib is None:
        raise RuntimeError('C MCI module not available')
    n = len(aom_for_ring)
    if n < 2:
        return 0.0
    m = int(np.asarray(aom_for_ring[0]).shape[0])
    stacked = _stack_aoms(aom_for_ring)

    if _is_ctypes:
        ring = _make_ring(n)
        stacked_ptr = stacked.ctypes.data_as(POINTER(c_double))
        return _call_ctypes_nosym(ring, n, stacked_ptr, n, m)
    else:
        return float(_lib.compute_mci_nosym(np.arange(n, dtype=np.int32), n, stacked, n, m))


def compute_mci_natorbs_sym(aom_for_ring, occ):
    """Compute MCI using natural orbitals (symmetric variant)."""
    if _lib is None:
        raise RuntimeError('C MCI module not available')
    n = len(aom_for_ring)
    if n < 2:
        return 0.0
    m = int(np.asarray(aom_for_ring[0]).shape[0])
    stacked = _stack_aoms(aom_for_ring)
    occ_f = np.asfortranarray(np.asarray(occ, dtype=np.float64))

    if _is_ctypes:
        ring = _make_ring(n)
        stacked_ptr = stacked.ctypes.data_as(POINTER(c_double))
        occ_ptr = occ_f.ravel(order='F').ctypes.data_as(POINTER(c_double))
        return _call_ctypes_natorb_sym(ring, n, stacked_ptr, n, m, occ_ptr)
    else:
        return float(_lib.compute_mci_natorbs_sym(np.arange(n, dtype=np.int32), n, stacked, n, m, occ_f))


def compute_mci_natorbs_nosym(aom_for_ring, occ):
    """Compute MCI using natural orbitals (non-symmetric variant)."""
    if _lib is None:
        raise RuntimeError('C MCI module not available')
    n = len(aom_for_ring)
    if n < 2:
        return 0.0
    m = int(np.asarray(aom_for_ring[0]).shape[0])
    stacked = _stack_aoms(aom_for_ring)
    occ_f = np.asfortranarray(np.asarray(occ, dtype=np.float64))

    if _is_ctypes:
        ring = _make_ring(n)
        stacked_ptr = stacked.ctypes.data_as(POINTER(c_double))
        occ_ptr = occ_f.ravel(order='F').ctypes.data_as(POINTER(c_double))
        return _call_ctypes_natorb_nosym(ring, n, stacked_ptr, n, m, occ_ptr)
    else:
        return float(_lib.compute_mci_natorbs_nosym(np.arange(n, dtype=np.int32), n, stacked, n, m, occ_f))


# Small diagnostic printing (optional) - disabled by default
# if _LIB_PATH is None:
#     print('C MCI library not found: using Python implementations for MCI')
# else:
#     try:
#         print(f'C MCI library loaded from: {_LIB_PATH}')
#     except Exception:
#         pass
