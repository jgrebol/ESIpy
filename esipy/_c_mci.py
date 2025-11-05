# Minimal C-backed MCI shim for esipy
# Exposes: has_c_module(), compute_mci_restricted_mulliken(aom_for_ring),
#          compute_mci_restricted_pruned(aom_for_ring), compute_mci_no_mulliken(aom_for_ring, occ),
#          compute_mci_no_pruned(aom_for_ring, occ)

import os
import ctypes
from ctypes import c_int, c_double, POINTER
import numpy as np

# Try to locate shared library in package directory
_here = os.path.dirname(__file__)
_LIB_PATH = None
_cands = [os.path.join(_here, 'libmci.so'), os.path.join(_here, 'libc_mci.so'), os.path.join(_here, 'libmci.dylib')]
_lib = None
for p in _cands:
    if os.path.exists(p):
        try:
            _lib = ctypes.CDLL(p)
            _LIB_PATH = p
            break
        except Exception:
            _lib = None
# If not found in package, try to load by name (system path / cwd)
if _lib is None:
    try:
        _lib = ctypes.CDLL('libmci.so')
        _LIB_PATH = 'libmci.so'
    except Exception:
        _lib = None

# Configure prototypes if library present
if _lib is not None:
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
        # If anything goes wrong, don't crash import; mark lib as unavailable
        _lib = None


def has_c_module():
    """Return True if the C shared library was found and loaded."""
    return _lib is not None


def _stack_aoms(aom_list):
    """Stack a list of (m,m) numpy arrays into a 1D column-major buffer expected by C.
    Each matrix is stored in column-major order consecutively: mat0, mat1, ...
    Returns a contiguous 1D np.float64 array.
    """
    if len(aom_list) == 0:
        raise ValueError('aom_list must be non-empty')
    m = int(np.asarray(aom_list[0]).shape[0])
    na = len(aom_list)
    stacked = np.empty((na * m * m,), dtype=np.float64)
    for i, A in enumerate(aom_list):
        Af = np.asfortranarray(np.asarray(A, dtype=np.float64))
        if Af.shape != (m, m):
            raise ValueError('All AOMs must be square with same shape')
        stacked[i * m * m : (i + 1) * m * m] = Af.ravel(order='F')
    return stacked


# Public API wrappers expected by indicators.py

def compute_mci_restricted_mulliken(aom_for_ring):
    """Compute MCI for restricted (no occ) Mulliken-like partition using the C backend.
    aom_for_ring: list/sequence of n numpy arrays (m x m) already ordered for the ring.
    Returns float.
    """
    if not has_c_module():
        raise RuntimeError('C MCI module not available')
    n = len(aom_for_ring)
    if n < 2:
        return 0.0
    m = int(np.asarray(aom_for_ring[0]).shape[0])
    ring = (c_int * n)(*range(n))  # 0..n-1 (aom_for_ring is already ordered)
    stacked = _stack_aoms(aom_for_ring)
    stacked_ptr = stacked.ctypes.data_as(POINTER(c_double))
    return float(_lib.compute_mci_sym(ring, c_int(n), stacked_ptr, c_int(n), c_int(m)))


def compute_mci_restricted_pruned(aom_for_ring):
    """Non-symmetric/pruned variant for restricted case."""
    if not has_c_module():
        raise RuntimeError('C MCI module not available')
    n = len(aom_for_ring)
    if n < 2:
        return 0.0
    m = int(np.asarray(aom_for_ring[0]).shape[0])
    ring = (c_int * n)(*range(n))
    stacked = _stack_aoms(aom_for_ring)
    stacked_ptr = stacked.ctypes.data_as(POINTER(c_double))
    return float(_lib.compute_mci_nosym(ring, c_int(n), stacked_ptr, c_int(n), c_int(m)))


def compute_mci_no_mulliken(aom_for_ring, occ):
    """Compute MCI for correlated (with occ) Mulliken variant."""
    if not has_c_module():
        raise RuntimeError('C MCI module not available')
    n = len(aom_for_ring)
    if n < 2:
        return 0.0
    m = int(np.asarray(aom_for_ring[0]).shape[0])
    ring = (c_int * n)(*range(n))
    stacked = _stack_aoms(aom_for_ring)
    stacked_ptr = stacked.ctypes.data_as(POINTER(c_double))
    occ_f = np.asfortranarray(np.asarray(occ, dtype=np.float64))
    occ_ptr = occ_f.ravel(order='F').ctypes.data_as(POINTER(c_double))
    return float(_lib.compute_mci_natorbs_sym(ring, c_int(n), stacked_ptr, c_int(n), c_int(m), occ_ptr))


def compute_mci_no_pruned(aom_for_ring, occ):
    """Compute MCI for correlated (with occ) pruned/non-symmetric variant."""
    if not has_c_module():
        raise RuntimeError('C MCI module not available')
    n = len(aom_for_ring)
    if n < 2:
        return 0.0
    m = int(np.asarray(aom_for_ring[0]).shape[0])
    ring = (c_int * n)(*range(n))
    stacked = _stack_aoms(aom_for_ring)
    stacked_ptr = stacked.ctypes.data_as(POINTER(c_double))
    occ_f = np.asfortranarray(np.asarray(occ, dtype=np.float64))
    occ_ptr = occ_f.ravel(order='F').ctypes.data_as(POINTER(c_double))
    return float(_lib.compute_mci_natorbs_nosym(ring, c_int(n), stacked_ptr, c_int(n), c_int(m), occ_ptr))

# Minimal convenience aliases matching previous naming used elsewhere (optional)
compute_mci_sym = compute_mci_restricted_mulliken
compute_mci_nosym = compute_mci_restricted_pruned
compute_mci_natorbs_sym = compute_mci_no_mulliken
compute_mci_natorbs_nosym = compute_mci_no_pruned

# Export the raw CDLL handle and print a one-line status message on import
_LIB_HANDLE = _lib
if _LIB_HANDLE is None:
    print('C MCI library not found: using Python implementations for MCI')
else:
    try:
        print(f'C MCI library loaded from: {_LIB_PATH}')
    except Exception:
        print('C MCI library loaded')
