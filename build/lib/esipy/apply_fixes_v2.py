
import os
import re

def fix_naming(path):
    tools_path = os.path.join(path, "esipy/tools.py")
    if os.path.exists(tools_path):
        with open(tools_path, "r") as f:
            content = f.read()
        content = content.replace('"meta_lowdin"', '"meta-lowdin"')
        with open(tools_path, "w") as f:
            f.write(content)

def fix_make_aoms(path):
    p = os.path.join(path, "esipy/make_aoms.py")
    if not os.path.exists(p): return
    with open(p, "r") as f:
        content = f.read()
    
    # Fix CASSCF None check and UnboundLocalError
    # Replace the Restricted block start
    old_restricted = """    # 2. RESTRICTED
    else:
        if hasattr(mf, 'mo_occ'):
            coeff = mf.mo_coeff[:, mf.mo_occ > 0]
        else:
            occ, coeff = get_natorbs(mf, S)
            coeff = coeff[:, occ > 1e-10]"""
    
    new_restricted = """    # 2. RESTRICTED
    else:
        if hasattr(mf, 'mo_occ') and mf.mo_occ is not None:
            coeff = mf.mo_coeff[:, mf.mo_occ > 0]
        elif hasattr(mf, 'mo_occ') and mf.mo_occ is None:
             # Fallback for CASSCF where mo_occ might be None in some PySCF versions/states
             coeff = mf.mo_coeff
        else:
            occ, coeff = get_natorbs(mf, S)
            coeff = coeff[:, occ > 1e-10]"""
    
    if old_restricted in content:
        content = content.replace(old_restricted, new_restricted)
    else:
        # Try a more generic regex if exact match fails due to whitespace
        content = re.sub(r"if hasattr\(mf, 'mo_occ'\):\s+coeff = mf\.mo_coeff\[:, mf\.mo_occ > 0\]", 
                         "if hasattr(mf, 'mo_occ') and mf.mo_occ is not None:\n            coeff = mf.mo_coeff[:, mf.mo_occ > 0]\n        elif hasattr(mf, 'mo_occ') and mf.mo_occ is None:\n            coeff = mf.mo_coeff", content)

    # Fix Unrestricted block
    content = re.sub(r"oa > 0", "oa is not None and oa > 0", content)
    content = re.sub(r"ob > 0", "ob is not None and ob > 0", content)

    with open(p, "w") as f:
        f.write(content)

def upgrade_qchem(path):
    # Copy modern tools.py, readfchk.py, make_aoms.py from main if they are too old
    # Actually, let's just apply the specific Q-Chem fixes to readfchk.py
    # and update tools.py signature.
    
    tools_path = os.path.join(path, "esipy/tools.py")
    with open(tools_path, "r") as f:
        content = f.read()
    
    if "use_sqrt_sii=False" not in content:
        content = content.replace("def permute_aos_rows(mat, mole2):", "def permute_aos_rows(mat, mole2, use_sqrt_sii=False):")
        # Add the scaling logic if missing
        scaling_logic = """    # 1/sqrt(Sii) scaling
    if use_sqrt_sii:
        s = mol.intor_symmetric('int1e_ovlp')
        s_diag = np.sqrt(np.diag(s))
        if mat.ndim == 2:
            if mat.shape[0] == len(s_diag): mat = mat * s_diag[:, None]
            if mat.shape[1] == len(s_diag): mat = mat * s_diag[None, :]
    """
        content = re.sub(r"mol = mole2\.pyscf_mol", "mol = mole2.pyscf_mol\n" + scaling_logic, content)
    
    with open(tools_path, "w") as f:
        f.write(content)

    readfchk_path = os.path.join(path, "esipy/readfchk.py")
    with open(readfchk_path, "r") as f:
        content = f.read()
    
    # Add is_qchem detection
    if "is_qchem" not in content:
        content = re.sub(r"self\.path = path", "self.path = path\n        with open(path, 'r') as f:\n            first_line = f.readline()\n            self.is_qchem = 'Q-Chem' in first_line or 'Q-CHEM' in first_line", content)
    
    # Add get_ovlp
    if "def get_ovlp(self):" not in content:
        get_ovlp_code = """    def get_ovlp(self):
        if hasattr(self.mole2.fchk, 'overlap_matrix') and self.mole2.fchk.overlap_matrix is not None:
            from esipy.tools import permute_aos_rows
            S = self.mole2.fchk.overlap_matrix
            S = permute_aos_rows(S, self.mole2, use_sqrt_sii=True)
            return S
        return self.mol.intor_symmetric('int1e_ovlp')
"""
        # Insert before another method or at the end of MeanField2
        content = re.sub(r"def make_rdm1\(self\):", get_ovlp_code + "\n    def make_rdm1(self):", content)

    # Add overlap_matrix reading to FchkMolecule
    if "overlap_matrix" not in content:
         # Find FchkMolecule.__init__
         content = re.sub(r"self\.e_tot = float\(read_from_fchk\('Total Energy', path\)\[-1\]\)", 
                          "try:\n            self.e_tot = float(read_from_fchk('Total Energy', path)[-1])\n        except:\n            self.e_tot = 0.0\n        self.overlap_matrix = None\n        if getattr(self, 'is_qchem', False):\n            ovlp_flat = read_list_from_fchk('Overlap Matrix', path)\n            if ovlp_flat:\n                nbasis = int(read_from_fchk('Number of basis functions', path)[-1])\n                if len(ovlp_flat) == nbasis * (nbasis + 1) // 2:\n                    self.overlap_matrix = np.zeros((nbasis, nbasis))\n                    idx = 0\n                    for i in range(nbasis):\n                        for j in range(i + 1):\n                            self.overlap_matrix[i, j] = self.overlap_matrix[j, i] = ovlp_flat[idx]\n                            idx += 1", content)

    with open(readfchk_path, "w") as f:
        f.write(content)

for b in ["main", "dev-iao", "dev-qchem2"]:
    path = f"../branch_verification/{b}"
    fix_naming(path)
    fix_make_aoms(path)
    if b == "dev-qchem2":
        upgrade_qchem(path)
