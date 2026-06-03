import numpy as np
from pyscf import gto
mol = gto.M(atom="C 0 0 0", basis="cc-pVTZ", cart=True).build()
s = mol.intor("int1e_ovlp")
print("PySCF F Norms (diag):", np.diag(s)[25:35])
