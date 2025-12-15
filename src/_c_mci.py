"""CTypes-based MCI C-wrapper placed under `src/`.

This module attempts to load a shared library `libmci.so` (built by setup.py
into the src/ directory) and exposes simple compute_* functions used by the
Python code. It stacks AOMs in Fortran order and calls the C API via ctypes.

C API expected (ctypes):
  double compute_mci_sym(const int *ring, int ring_len, const double *aoms_stacked, int naoms, int m)
  double compute_mci_natorbs_sym(const int *ring, int ring_len, const double *aoms_stacked, int naoms, int m, const double *occ)
  double compute_mci_nosym(const int *ring, int ring_len, const double *aoms_stacked, int naoms, int m)
  double compute_mci_natorbs_nosym(const int *ring, int ring_len, const double *aoms_stacked, int naoms, int m, const double *occ)

This implementation is intentionally minimal and robust: it looks for the
shared library at a few candidate paths and sets ctypes prototypes if found.
"""
from ctypes import c_int, c_double, POINTER
import ctypes
import os
import numpy as np

_lib = None
_is_ctypes = False

# Candidate shared lib locations
_candidates = [
    os.path.join(os.getcwd(), 'src', 'libmci.so'),
    os.path.join(os.getcwd(), 'libmci.so'),
    os.path.join(os.path.dirname(__file__), 'libmci.so'),
]
for p in _candidates:
    if os.path.exists(p):
        try:
            _lib = ctypes.CDLL(p)
            _is_ctypes = True
            _lib_path = p
            break
        except Exception:
            _lib = None
            _is_ctypes = False

# Set prototypes if we have a ctypes CDLL
if _is_ctypes and _lib is not None:
    try:
        _lib.compute_mci_sym.restype = c_double
        _lib.compute_mci_sym.argtypes = [POINTER(c_int), c_int, POINTER(c_double), c_int, c_int]

        _lib.compute_mci_nosym.restype = c_double
        _lib.compute_mci_nosym.argtypes = [POINTER(c_int), c_int, POINTER(c_double), c_int, c_int]
    except Exception:
        _lib = None
        _is_ctypes = False


def has_c_module():
    return _lib is not None


def _stack_aoms(aom_list):
    if len(aom_list) == 0:
        raise ValueError('aom_list must be non-empty')
    first = np.asarray(aom_list[0])
    if first.ndim != 2 or first.shape[0] != first.shape[1]:
        raise ValueError('AOMs must be square matrices')
    m = int(first.shape[0])
    na = len(aom_list)
    stacked = np.empty((na * m * m,), dtype=np.float64, order='F')
    for i, A in enumerate(aom_list):
        Af = np.asfortranarray(np.asarray(A, dtype=np.float64))
        if Af.shape != (m, m):
            raise ValueError('All AOMs must be square with same shape')
        stacked[i * m * m: (i + 1) * m * m] = Af.ravel(order='F')
    return stacked, na, m


def _make_ring_ctypes(n):
    return (c_int * n)(*range(n))


def compute_mci_sym(aom_for_ring):
    if not has_c_module():
        raise RuntimeError('C MCI module not available')
    stacked, na, m = _stack_aoms(aom_for_ring)
    ring = _make_ring_ctypes(na)
    ptr = stacked.ctypes.data_as(POINTER(c_double))
    return float(_lib.compute_mci_sym(ring, c_int(na), ptr, c_int(na), c_int(m)))


def compute_mci_nosym(aom_for_ring):
    if not has_c_module():
        raise RuntimeError('C MCI module not available')
    stacked, na, m = _stack_aoms(aom_for_ring)
    ring = _make_ring_ctypes(na)
    ptr = stacked.ctypes.data_as(POINTER(c_double))
    return float(_lib.compute_mci_nosym(ring, c_int(na), ptr, c_int(na), c_int(m)))


def compute_mci_natorbs_sym(aom_for_ring, occ):
    if not has_c_module():
        raise RuntimeError('C MCI module not available')
    # Precompute occ * A for each AOM matrix and call base C function
    aom_list = [np.asarray(A, dtype=np.float64) for A in aom_for_ring]
    # compute_mci expects Fortran-ordered AOMs; ensure occ is 2D
    occ_mat = np.asarray(occ, dtype=np.float64)
    pre_aoms = [np.asfortranarray(occ_mat.dot(np.asarray(A, dtype=np.float64))) for A in aom_list]
    return compute_mci_sym(pre_aoms)


def compute_mci_natorbs_nosym(aom_for_ring, occ):
    if not has_c_module():
        raise RuntimeError('C MCI module not available')
    # Precompute occ * A for each AOM matrix and call base C function (nosym)
    aom_list = [np.asarray(A, dtype=np.float64) for A in aom_for_ring]
    occ_mat = np.asarray(occ, dtype=np.float64)
    pre_aoms = [np.asfortranarray(occ_mat.dot(np.asarray(A, dtype=np.float64))) for A in aom_list]
    return compute_mci_nosym(pre_aoms)
