import numpy as np
import ctypes
import os
from collections import OrderedDict
import threading

# Load the shared library from the package directory (more robust than cwd)
_lib_path = os.path.join(os.path.dirname(__file__), 'libmci.so')
lib = ctypes.cdll.LoadLibrary(os.path.abspath(_lib_path))

# function prototypes
lib.compute_mci_sym.restype = ctypes.c_double
lib.compute_mci_sym.argtypes = [
    ctypes.POINTER(ctypes.c_int), ctypes.c_int,
    ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int
]

lib.compute_mci_natorbs_sym.restype = ctypes.c_double
lib.compute_mci_natorbs_sym.argtypes = [
    ctypes.POINTER(ctypes.c_int), ctypes.c_int,
    ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_double)
]

lib.compute_mci_nosym.restype = ctypes.c_double
lib.compute_mci_nosym.argtypes = [
    ctypes.POINTER(ctypes.c_int), ctypes.c_int,
    ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int
]

lib.compute_mci_natorbs_nosym.restype = ctypes.c_double
lib.compute_mci_natorbs_nosym.argtypes = [
    ctypes.POINTER(ctypes.c_int), ctypes.c_int,
    ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_double)
]


class MCIWrapper:
    """Wrapper that centralizes input preparation and provides a small LRU cache for stacked AOM arrays.

    Goals and guarantees:
    - Reuse stacked AOM buffers when the same arrays (by identity) are supplied repeatedly.
    - Preserve existing function names and signatures by exposing module-level helpers.
    - Keep changes minimal and safe: no change to C interface signatures or math.
    """

    def __init__(self, lib=lib, max_cache=128):
        self.lib = lib
        # simple LRU cache mapping key -> stacked 1D Fortran-ordered array
        self._stack_cache = OrderedDict()
        self._max_cache = int(max_cache)
        # protect cache for concurrent access
        self._lock = threading.RLock()

    def _stack_key(self, aom_list):
        # Key uses object identity and shape/dtype to be cheap.
        # Assumption: callers will typically reuse the same numpy arrays when computing many rings.
        return tuple((id(A), A.shape, str(A.dtype)) for A in aom_list)

    def _stack_aoms(self, aom_list):
        """Stack list of (m,m) matrices into a single 1D Fortran-ordered array (column-major).

        Uses a small LRU cache keyed by array identity and shape/dtype to avoid repeated copies when
        the same arrays are reused by the caller. This keeps the change minimal and cheap.
        """
        if not aom_list:
            raise ValueError("aom_list must be a non-empty list of square matrices")

        key = self._stack_key(aom_list)
        with self._lock:
            cached = self._stack_cache.get(key)
            if cached is not None:
                # move to end to mark as recently used
                self._stack_cache.move_to_end(key)
                return cached

        m = aom_list[0].shape[0]
        na = len(aom_list)
        # ensure Fortran contiguous and dtype double
        arrs = [np.asfortranarray(np.asarray(A, dtype=np.float64)) for A in aom_list]
        stacked = np.empty((na * m * m,), dtype=np.float64, order='C')
        for i, A in enumerate(arrs):
            stacked[i*m*m:(i+1)*m*m] = A.ravel(order='F')

        # cache it
        with self._lock:
            self._stack_cache[key] = stacked
            if len(self._stack_cache) > self._max_cache:
                # pop least recently used
                self._stack_cache.popitem(last=False)
        return stacked

    def _prepare_ring(self, ring):
        return np.asarray(ring, dtype=np.int32)

    def _prepare_occ(self, occ):
        return np.asfortranarray(np.asarray(occ, dtype=np.float64))

    def call_compute_mci_sym(self, ring, aom_list):
        ring_arr = self._prepare_ring(ring)
        stacked = self._stack_aoms(aom_list)
        n = ring_arr.size
        m = aom_list[0].shape[0]
        na = len(aom_list)
        return self.lib.compute_mci_sym(ring_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), ctypes.c_int(n),
                                        stacked.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ctypes.c_int(na), ctypes.c_int(m))

    def call_compute_mci_natorbs_sym(self, ring, aom_list, occ):
        ring_arr = self._prepare_ring(ring)
        stacked = self._stack_aoms(aom_list)
        occ_f = self._prepare_occ(occ)
        n = ring_arr.size
        m = aom_list[0].shape[0]
        na = len(aom_list)
        return self.lib.compute_mci_natorbs_sym(ring_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), ctypes.c_int(n),
                                                stacked.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ctypes.c_int(na), ctypes.c_int(m),
                                                occ_f.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

    def call_compute_mci_nosym(self, ring, aom_list):
        ring_arr = self._prepare_ring(ring)
        stacked = self._stack_aoms(aom_list)
        n = ring_arr.size
        m = aom_list[0].shape[0]
        na = len(aom_list)
        return self.lib.compute_mci_nosym(ring_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), ctypes.c_int(n),
                                          stacked.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ctypes.c_int(na), ctypes.c_int(m))

    def call_compute_mci_natorbs_nosym(self, ring, aom_list, occ):
        ring_arr = self._prepare_ring(ring)
        stacked = self._stack_aoms(aom_list)
        occ_f = self._prepare_occ(occ)
        n = ring_arr.size
        m = aom_list[0].shape[0]
        na = len(aom_list)
        return self.lib.compute_mci_natorbs_nosym(ring_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), ctypes.c_int(n),
                                                  stacked.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ctypes.c_int(na), ctypes.c_int(m),
                                                  occ_f.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

    # --- small cache management helpers ---
    def clear_stack_cache(self):
        """Clear the internal stacked-AOM LRU cache."""
        with self._lock:
            self._stack_cache.clear()

    def stack_cache_info(self):
        """Return simple cache info: current size and max size."""
        with self._lock:
            return {"size": len(self._stack_cache), "max": self._max_cache}

    def set_stack_cache_size(self, new_max):
        """Adjust the LRU cache max size; evict if current size exceeds new max."""
        with self._lock:
            self._max_cache = int(new_max)
            while len(self._stack_cache) > self._max_cache:
                self._stack_cache.popitem(last=False)


# Module-level default wrapper to preserve original convenience functions
_default_wrapper = MCIWrapper()

# Expose original-style functions for backward compatibility
def stack_aoms(aom_list):
    return _default_wrapper._stack_aoms(aom_list)


def call_compute_mci_sym(ring, aom_list):
    return _default_wrapper.call_compute_mci_sym(ring, aom_list)


def call_compute_mci_natorbs_sym(ring, aom_list, occ):
    return _default_wrapper.call_compute_mci_natorbs_sym(ring, aom_list, occ)


def call_compute_mci_nosym(ring, aom_list):
    return _default_wrapper.call_compute_mci_nosym(ring, aom_list)


def call_compute_mci_natorbs_nosym(ring, aom_list, occ):
    return _default_wrapper.call_compute_mci_natorbs_nosym(ring, aom_list, occ)

# Expose cache-management helpers at module level
def clear_stack_cache():
    return _default_wrapper.clear_stack_cache()


def stack_cache_info():
    return _default_wrapper.stack_cache_info()


def set_stack_cache_size(new_max):
    return _default_wrapper.set_stack_cache_size(new_max)
