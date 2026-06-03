import numpy as np
from esipy.readfchk import readfchk
from pyscf import gto

mol, mf = readfchk("../tests/FCHK/GAUSSIAN/3_o2_triplet.fchk")
s = mol.intor_symmetric('int1e_ovlp')
ca, cb = mf.mo_coeff

def check_ortho(c, label):
    ovlp = c.T @ s @ c
    diag = np.diag(ovlp)
    print(f"--- {label} ---")
    print(f"  Diag (first 10): {np.round(diag[:10], 4)}")
    print(f"  Max off-diag: {np.max(np.abs(ovlp - np.diag(diag))):.4e}")
    # Find which MO is not normalized
    bad_idx = np.where(np.abs(diag - 1.0) > 1e-3)[0]
    if len(bad_idx) > 0:
        print(f"  Bad MO indices: {bad_idx}")
        print(f"  Values: {diag[bad_idx]}")

check_ortho(ca, "Alpha")
check_ortho(cb, "Beta")

# Check SP shells specifically
print("\nBasis labels:")
for i, l in enumerate(mol.pyscf_mol.ao_labels()):
    print(f"  {i}: {l}")
