
from esipy.readfchk import readfchk
path = '/home/joan/PycharmProjects/ESIpy/joan/bzt.hs'
try:
    mol, mf = readfchk(path)
    print("Successfully read FCHK")
    print("Wavefunction type:", "Unrestricted" if getattr(mf, "unrestricted", False) or mf.__class__.__name__ == "UHF" else "Restricted")
    print("MO Coeff shape:", [np.shape(c) for c in mf.mo_coeff] if isinstance(mf.mo_coeff, (list, tuple)) else np.shape(mf.mo_coeff))
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
