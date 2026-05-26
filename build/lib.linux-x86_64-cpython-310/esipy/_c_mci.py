# Bridge wrapper: re-export implementation from src/_c_mci.py when present.
import os
import importlib.util
from typing import Sequence

_impl = None
_impl_path = os.path.join(os.getcwd(), 'src', '_c_mci.py')
if os.path.exists(_impl_path):
    spec = importlib.util.spec_from_file_location('src._c_mci', _impl_path)
    if spec and spec.loader:
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)  # type: ignore
            _impl = mod
        except Exception:
            _impl = None


def has_c_module() -> bool:
    return _impl is not None and getattr(_impl, 'has_c_module', lambda: False)()


def compute_mci_sym(aom_for_ring: Sequence):
    if not has_c_module():
        raise RuntimeError('C MCI module not available')
    return _impl.compute_mci_sym(aom_for_ring)


def compute_mci_nosym(aom_for_ring: Sequence):
    if not has_c_module():
        raise RuntimeError('C MCI module not available')
    return _impl.compute_mci_nosym(aom_for_ring)


def compute_mci_natorbs_sym(aom_for_ring, occ):
    if not has_c_module():
        raise RuntimeError('C MCI module not available')
    return _impl.compute_mci_natorbs_sym(aom_for_ring, occ)


def compute_mci_natorbs_nosym(aom_for_ring, occ):
    if not has_c_module():
        raise RuntimeError('C MCI module not available')
    return _impl.compute_mci_natorbs_nosym(aom_for_ring, occ)
