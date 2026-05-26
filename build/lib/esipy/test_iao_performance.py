
import time
import numpy as np
from pyscf import gto, scf
from esipy.iao import iao

def test():
    mol = gto.M(atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587', basis='cc-pVTZ')
    mf = scf.RHF(mol)
    mf.kernel()
    
    start = time.time()
    C_iao, pmol = iao(mol, mf.mo_coeff[:, mf.mo_occ > 0])
    end = time.time()
    print(f"IAO time: {end - start:.4f} s")
    print(f"pmol.nao: {pmol.nao}")
    
    # Check some property
    S = mol.intor('int1e_ovlp')
    P_AO = mf.make_rdm1()
    # Construct AOMs manually for O
    ao_loc_pmol = pmol.ao_loc_nr()
    U = S @ C_iao
    # O is atom 0
    v_o = C_iao.T @ S[:, ao_loc_pmol[0]:ao_loc_pmol[1]]
    # Wait, IAO AOM construction is different.
    # In make_aoms.py:
    # U = np.dot(S, U_nonorth)
    # return [np.linalg.multi_dot((c.T, U, eta[i], U.T, c)) for i in range(mol.natm)]
    
    # Just print first few elements of C_iao
    print("C_iao[:5, :5]:")
    print(C_iao[:5, :5])

if __name__ == "__main__":
    test()
