
import os

def fix_common(path):
    # Fix meta-lowdin naming
    tools_path = os.path.join(path, "esipy/tools.py")
    if os.path.exists(tools_path):
        with open(tools_path, "r") as f:
            content = f.read()
        content = content.replace('"meta_lowdin"', '"meta-lowdin"')
        with open(tools_path, "w") as f:
            f.write(content)
    
    # Fix CASSCF NoneType error in make_aoms.py
    make_aoms_path = os.path.join(path, "esipy/make_aoms.py")
    if os.path.exists(make_aoms_path):
        with open(make_aoms_path, "r") as f:
            content = f.read()
        # More robust fix for CASSCF
        import re
        content = re.sub(r"oa > 0", "oa is not None and oa > 0", content)
        content = re.sub(r"ob > 0", "ob is not None and ob > 0", content)
        content = re.sub(r"occ > 0", "occ is not None and occ > 0", content)
        with open(make_aoms_path, "w") as f:
            f.write(content)

def fix_qchem(path):
    # Add Q-Chem support to readfchk.py
    readfchk_path = os.path.join(path, "esipy/readfchk.py")
    if os.path.exists(readfchk_path):
        with open(readfchk_path, "r") as f:
            lines = f.readlines()
        
        new_lines = []
        for line in lines:
            if "self.path = path" in line and "class FchkMolecule" in lines[lines.index(line)-1]:
                new_lines.append(line)
                new_lines.append("        with open(path, 'r') as f:\n")
                new_lines.append("            first_line = f.readline()\n")
                new_lines.append("            self.is_qchem = \"Q-Chem\" in first_line or \"Q-CHEM\" in first_line\n")
            elif "self.e_tot = float(read_from_fchk('Total Energy', path)[-1])" in line:
                new_lines.append("        try:\n")
                new_lines.append("            self.e_tot = float(read_from_fchk('Total Energy', path)[-1])\n")
                new_lines.append("        except:\n")
                new_lines.append("            self.e_tot = 0.0\n")
            elif "self.overlap_matrix = None" in line:
                 new_lines.append(line)
                 new_lines.append("        if getattr(self, 'is_qchem', False):\n")
                 new_lines.append("            ovlp_flat = read_list_from_fchk('Overlap Matrix', path)\n")
                 new_lines.append("            if ovlp_flat:\n")
                 new_lines.append("                nbasis = int(read_from_fchk('Number of basis functions', path)[-1])\n")
                 new_lines.append("                if len(ovlp_flat) == nbasis * (nbasis + 1) // 2:\n")
                 new_lines.append("                    self.overlap_matrix = np.zeros((nbasis, nbasis))\n")
                 new_lines.append("                    idx = 0\n")
                 new_lines.append("                    for i in range(nbasis):\n")
                 new_lines.append("                        for j in range(i + 1):\n")
                 new_lines.append("                            self.overlap_matrix[i, j] = self.overlap_matrix[j, i] = ovlp_flat[idx]\n")
                 new_lines.append("                            idx += 1\n")
            else:
                new_lines.append(line)
        
        with open(readfchk_path, "w") as f:
            f.writelines(new_lines)

for b in ["main", "dev-iao", "dev-qchem2"]:
    path = f"../branch_verification/{b}"
    fix_common(path)
    if b == "dev-qchem2":
        fix_qchem(path)
