import re

def fix_input():
    with open('input.py', 'r') as f: content = f.read()
    if 'self.ecp = None' not in content:
        content = content.replace('self.fchk_file = None', 'self.fchk_file = None\n        self.ecp = None')
        content = content.replace("elif line.startswith('$READAOM'):", "elif line.startswith('$ECP'):\n                i += 1\n                if i < len(lines): obj.ecp = lines[i].strip()\n            elif line.startswith('$READAOM'):")
        with open('input.py', 'w') as f: f.write(content)

def fix_readfchk():
    with open('readfchk.py', 'r') as f: content = f.read()
    # 1. Add ecp support to FchkMolecule
    if 'self.nuclear_charges = None' not in content:
        content = content.replace('self.atomic_numbers = [int(i) for i in read_list_from_fchk', 
                                  'res_nuc = read_list_from_fchk("Nuclear charges", path)\n        self.nuclear_charges = [float(i) for i in res_nuc] if res_nuc else None\n        self.atomic_numbers = [int(i) for i in read_list_from_fchk')
    
    # 2. Add ecp support and error check to Mole2
    if 'self.ecp = None' not in content:
        content = content.replace('self.charge = int(getattr(self.fchk, \'charge\', 0))', 
                                  'self.charge = int(getattr(self.fchk, \'charge\', 0))\n        self.ecp = None\n        self.nuclear_charges = getattr(self.fchk, \'nuclear_charges\', None)')
        
        # Check for missing ECP in build
        ecp_check = """
        if self.ecp is None and self.fchk.nuclear_charges is not None:
             # Basic check: if Z != Z_eff, we likely have an ECP
             if not np.allclose(self.fchk.atomic_numbers, self.fchk.nuclear_charges):
                 raise RuntimeError("Calculation requires ecp, but no ECP is defined")
        self.pyscf_mol.ecp = self.ecp
"""
        content = content.replace('self.pyscf_mol.unit = \'Bohr\'', 'self.pyscf_mol.unit = \'Bohr\'\n' + ecp_check)
    
    with open('readfchk.py', 'w') as f: f.write(content)

if __name__ == "__main__":
    fix_input()
    fix_readfchk()
