import os
import ctypes
from ctypes import c_int, c_double, POINTER
import numpy as np

_lib = None
_LIB_PATH = None

# Try to import compiled extension (esipy/libmci.*)
try:
    import esipy.mci as libmci_ext  # type: ignore
    _lib = libmci_ext
    _LIB_PATH = getattr(libmci_ext, '__file__', None)
except Exception:
    _lib = None
    _LIB_PATH = None

# If not, try to find shared object in package dir
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

# 3) If loaded a ctypes CDLL, set the prototypes for the functions we expect
if _lib is not None and isinstance(_lib, ctypes.CDLL):
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
        # If prototypes fail to set, discard the library reference to force Python fallback
        _lib = None


def has_c_module():
    return _lib is not None


# Stack helper for C implementation
def _stack_aoms(aom_list):
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

# MCI for symmetric AOMs (Mulliken and QTAIM)
def compute_mci_nosym(aom_for_ring):
    if _lib is None:
        raise RuntimeError('C MCI module not available')
    n = len(aom_for_ring)
    if n < 2:
        return 0.0
    m = int(np.asarray(aom_for_ring[0]).shape[0])
    ring = (c_int * n)(*range(n))
    stacked = _stack_aoms(aom_for_ring)
    stacked_ptr = stacked.ctypes.data_as(POINTER(c_double))
    if isinstance(_lib, ctypes.CDLL):
        # call the non-symmetric C symbol
        return float(_lib.compute_mci_nosym(ring, c_int(n), stacked_ptr, c_int(n), c_int(m)))
    else:
        # if imported as Python extension that exposes C-callable functions
        return float(_lib.compute_mci_sym(ring, n, stacked, n, m))


# MCI for symmetric AOMs
def compute_mci_sym(aom_for_ring):
    if _lib is None:
        raise RuntimeError('C MCI module not available')
    n = len(aom_for_ring)
    if n < 2:
        return 0.0
    m = int(np.asarray(aom_for_ring[0]).shape[0])
    ring = (c_int * n)(*range(n))
    stacked = _stack_aoms(aom_for_ring)
    stacked_ptr = stacked.ctypes.data_as(POINTER(c_double))
    if isinstance(_lib, ctypes.CDLL):
        return float(_lib.compute_mci_sym(ring, c_int(n), stacked_ptr, c_int(n), c_int(m)))
    else:
        return float(_lib.compute_mci_nosym(ring, n, stacked, n, m))


# MCI with natural orbitals for symmetric AOMs (Mulliken and QTAIM)
def compute_mci_natorbs_sym(aom_for_ring, occ):
    if _lib is None:
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
    if isinstance(_lib, ctypes.CDLL):
        return float(_lib.compute_mci_natorbs_sym(ring, c_int(n), stacked_ptr, c_int(n), c_int(m), occ_ptr))
    else:
        return float(_lib.compute_mci_natorbs_sym(ring, n, stacked, n, m, occ))


# MCI with natural orbitals for non-symmetric AOMs
def compute_mci_natorbs_nosym(aom_for_ring, occ):
    if _lib is None:
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
    if isinstance(_lib, ctypes.CDLL):
        return float(_lib.compute_mci_natorbs_nosym(ring, c_int(n), stacked_ptr, c_int(n), c_int(m), occ_ptr))
    else:
        return float(_lib.compute_mci_natorbs_nosym(ring, n, stacked, n, m, occ))


# Export libhandle info
#_LIB_HANDLE = _lib
#if _LIB_HANDLE is None:
#    print('C MCI library not found: using Python implementations for MCI')
#else:
#    try:
#        print(f'C MCI library loaded from: {_LIB_PATH}')
#    except Exception:
#        print('C MCI library loaded')
