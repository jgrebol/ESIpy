
import os
import numpy as np
from readfchk import readfchk
from tools import find_ns, wf_type

def test_fchk(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
    
    print(f"\n--- Testing: {path} ---")
    mol, mf = readfchk(path)
    dm = mf.make_rdm1()
    
    # Check electrons
    s = mf.get_ovlp()
    if isinstance(dm, np.ndarray) and dm.ndim == 3: # UHF
        nelec = np.trace(dm[0] @ s) + np.trace(dm[1] @ s)
    else:
        nelec = np.trace(dm @ s)
    print(f"Total electrons (from DM trace): {nelec:.4f}")
    print(f"Expected electrons: {mol.nelectron}")
    
    # Test find_ns
    from esipy.make_aoms import make_aoms
    # We need AOMs to test find_ns as it's currently defined
    # But wait, find_ns in tools.py expects a list of matrices
    # Let's just mock aom for a moment or use make_aoms if possible
    
    # Actually, let's just test if wf_type works on mf
    # Wait, indicators.py usually receives AOMs, not mf.
    
    print(f"Wavefunction type: {mf.__name__}")

if __name__ == "__main__":
    test_fchk("../FCHK/GAUSSIAN/bz.fchk")
    test_fchk("../FCHK/GAUSSIAN/unrest.fchk")
