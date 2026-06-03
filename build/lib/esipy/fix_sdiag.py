import os
import re

def fix_readfchk():
    with open('readfchk.py', 'r') as f:
        content = f.read()

    # Define Sdiag constants
    sdiag_defs = """
        # Get normalization factors from the raw overlap matrix
        from esipy.tools import permute_aos_rows
        
        is_qchem = getattr(self.mole2.fchk, 'is_qchem', False)
        
        if not is_qchem:
            # Standard Gaussian Sdiag factors (Gau/PySCF convention)
            SQRT3 = 1.7320508075688772
            SQRT5 = 2.2360679774997897
            
            sdiag_list = []
            shell_types = self.mole2.fchk.mssh
            for st in shell_types:
                if st == 0: sdiag_list.extend([1.0])
                elif st == 1: sdiag_list.extend([1.0]*3)
                elif st == -1: sdiag_list.extend([1.0]*4)
                elif st == 2: # 6D: XX, YY, ZZ, XY, XZ, YZ
                    sdiag_list.extend([1.0, 1.0, 1.0, SQRT3, SQRT3, SQRT3])
                elif st == 3: # 10F: XXX, YYY, ZZZ, XYY, XXY, XXZ, XZZ, YZZ, YYZ, XYZ
                    sdiag_list.extend([1.0, 1.0, 1.0, SQRT3, SQRT3, SQRT3, SQRT3, SQRT3, SQRT3, SQRT3*SQRT5])
                else:
                    l = abs(st)
                    n = (l+1)*(l+2)//2 if st > 0 else (2*l+1)
                    sdiag_list.extend([1.0]*n)
            v_fixed = np.array(sdiag_list)
        else:
            v_fixed = np.ones(self.nao)

        # S_raw is computed by PySCF using the reconstructed FCHK basis
        S_raw = self.mol.intor_symmetric('int1e_ovlp')
        v = 1.0 / np.sqrt(np.abs(np.diag(S_raw)))

        def standardize_mat(mat):
            # 1. Scale by Gaussian Sdiag BEFORE permutation
            if not is_qchem:
                if mat.ndim == 2:
                    if mat.shape[0] == self.nao: # MO coeffs (AO, MO)
                        mat = (mat.T / v_fixed).T
                    else: # Matrix (AO, AO)
                        mat = mat / np.outer(v_fixed, v_fixed)
            
            # 2. Reorder to PySCF shell layout
            mat_p = permute_aos_rows(mat, self.mole2)
            if mat_p.shape == mat_p.T.shape and mat_p.shape[0] == self.nao: # Matrix
                mat_p = permute_aos_rows(mat_p.T, self.mole2).T
                # 3. Scale to Unit-Normalization (PySCF convention)
                return mat_p * np.outer(v, v)
            else: # MO Coefficients
                return (mat_p.T * v).T
"""

    pattern = re.compile(r"(\s+)# Get normalization factors from the raw overlap matrix.*?return mat_p \* v\[:, None\]", re.DOTALL)
    
    match = pattern.search(content)
    if match:
        content = content[:match.start()] + sdiag_defs + content[match.end():]

    with open('readfchk.py', 'w') as f:
        f.write(content)

if __name__ == "__main__":
    fix_readfchk()
