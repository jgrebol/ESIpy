import numpy as np
from scipy.special import factorial

def readfchk(filename):
    """
    Reads the information from a fchk file.
    Args:
        filename: Name of the fchk file.

    Returns:
        Mole object and MeanField object.
    """
    mol = Mole(filename)
    mf = MeanField(filename, mol)
    return mol, mf

def read_from_fchk(to_read, path):
    with open(path, 'r') as f:
        for line in f:
            if line.startswith(to_read):
                return line.split()

def read_level_theory(path):
    with open(path, 'r') as f:
        next(f)
        return next(f).split()

def read_contract_coeff(path):
    l = []
    with open(path, 'r') as f:
        found = False
        for line in f:
            if found:
                break
            if line.startswith('Contraction coefficients'):
                s = int(line.split()[-1])
                found = True
                if found:
                    for num_str in line.split():
                        num = float(num_str)
                        if len(l) < s:
                            l.append(num)
                        else:
                            break
    return l


def read_list_from_fchk(start, end, path):
    l = []
    with open(path, 'r') as f:
        found = False
        for line in f:
            if line.startswith(start):
                found = True
                continue
            if end in line:
                break
            if found:
                l.extend(map(float, line.split()))
    return l

def read_atomic_symbols(z):
    z_to_symbol = {
        1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne',
        11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca',
        21: 'Sc', 22: 'Ti', 23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn',
        31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr', 37: 'Rb', 38: 'Sr', 39: 'Y', 40: 'Zr',
        41: 'Nb', 42: 'Mo', 43: 'Tc', 44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd', 49: 'In', 50: 'Sn',
        51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe', 55: 'Cs', 56: 'Ba', 57: 'La', 58: 'Ce', 59: 'Pr', 60: 'Nd',
        61: 'Pm', 62: 'Sm', 63: 'Eu', 64: 'Gd', 65: 'Tb', 66: 'Dy', 67: 'Ho', 68: 'Er', 69: 'Tm', 70: 'Yb',
        71: 'Lu', 72: 'Hf', 73: 'Ta', 74: 'W', 75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt', 79: 'Au', 80: 'Hg',
        81: 'Tl', 82: 'Pb', 83: 'Bi', 84: 'Po', 85: 'At', 86: 'Rn', 87: 'Fr', 88: 'Ra', 89: 'Ac', 90: 'Th',
        91: 'Pa', 92: 'U', 93: 'Np', 94: 'Pu', 95: 'Am', 96: 'Cm', 97: 'Bk', 98: 'Cf', 99: 'Es', 100: 'Fm',
        101: 'Md', 102: 'No', 103: 'Lr', 104: 'Rf', 105: 'Db', 106: 'Sg', 107: 'Bh', 108: 'Hs', 109: 'Mt',
        110: 'Ds', 111: 'Rg', 112: 'Cn', 113: 'Nh', 114: 'Fl', 115: 'Mc', 116: 'Lv', 117: 'Ts', 118: 'Og'
    }

    return [z_to_symbol[int(i)] for i in z]


def make_env(mol):
    """
    Build the _env array as PySCF does, using data from FCHK.
    One block of basis data per unique element.
    """
    from pyscf.gto.mole import gto_norm, _nomalize_contracted_ao

    # 1. Start with 20 zero-padding floats
    env = [0.0] * 20

    # 2. Append each atom's coordinates (in Bohr) plus a 0.0 zeta
    coords = mol.atom_coords()  # shape (natoms,3)
    for (x,y,z) in coords:
        env.extend([x, y, z, 0.0])

    # 3. Identify one representative atom for each element type
    uniques = {}
    for i in range(mol.natm):
        Z = mol.atomic_numbers[i]
        if Z not in uniques:
            uniques[Z] = i

    # Fix shell-to-atom indexing from FCHK (1->0-based)
    iatsh = [i-1 for i in mol.iatsh]

    # 4. Compute pointer array for primitives
    prim_ptr = [0] * (mol.ncshell + 1)
    for i in range(mol.ncshell):
        prim_ptr[i+1] = prim_ptr[i] + mol.mnsh[i]

    done_exps = set()

    # sort uniques
    uniques = {k: v for k, v in sorted(uniques.items(), key=lambda item: item[0])}
    # 5. Process shells for each unique atom type
    for rep_atom_idx in uniques.values():
        for shell_id in range(mol.ncshell):
            if iatsh[shell_id] != rep_atom_idx:
                continue
            l = mol.mssh[shell_id]
            nprim = mol.mnsh[shell_id]
            p0 = prim_ptr[shell_id]
            p1 = prim_ptr[shell_id+1]
            exps = np.array(mol.expsh[p0:p1])

            # Normalize and add the S&P parts if needed
            if l == -1:
                # SP shell: split into S(l=0) and P(l=1)
                # Determine number of contractions for S and P
                nctr_s = (len(mol.c1[p0:p1]) // nprim)
                nctr_p = (len(mol.c2[p0:p1]) // nprim)
                cs = np.array(mol.c1[p0:p1]).reshape(nprim, nctr_s)
                cp = np.array(mol.c2[p0:p1]).reshape(nprim, nctr_p)
                # S part
                cs = cs * gto_norm(0, exps).reshape(-1,1)
                cs = _nomalize_contracted_ao(0, exps, cs)
                env.extend(exps.tolist())
                env.extend(cs.T.reshape(-1).tolist())
                # P part
                cp = cp * gto_norm(1, exps).reshape(-1,1)
                cp = _nomalize_contracted_ao(1, exps, cp)
                env.extend(exps.tolist())
                env.extend(cp.T.reshape(-1).tolist())
            else:
                # Normal shell (s,p,d,f)
                # For cartesian d or f (l=-2,-3), treat as l=2,3
                l0 = abs(l)
                coeffs = mol.c1[p0:p1]
                nctr = len(coeffs) // nprim
                cs = np.array(coeffs).reshape(nprim, nctr)
                cs = cs * gto_norm(l0, exps).reshape(-1,1)
                cs = _nomalize_contracted_ao(l0, exps, cs)
                expvals = tuple(np.round(exps, 8))
                if expvals not in done_exps:
                    env.extend(exps.tolist())
                    done_exps.add(expvals)
                env.extend(cs.T.reshape(-1).tolist())

    return np.array(env, dtype=np.float32)

def make_atm(mol):
    """Construct _atm array and _env from Mole object manually."""
    _atm = []
    _env = []
    for i, (Z, coord) in enumerate(zip(mol.atomic_numbers, mol.atom_coords())):
        x, y, z = coord
        ptr_coord = len(_env) + 20  # Points to x
        _env.extend([x, y, z])
        ptr_zeta = len(_env) + 20  # Points to zeta
        _env.append(0.0)  # Nuclear zeta (0.0 for point charge)

        # 13-slot _atm record
        atm_rec = [0] * 6
        atm_rec[0] = Z                   # Atomic number
        atm_rec[1] = ptr_coord          # Pointer to coordinates in _env
        atm_rec[2] = 1                   # Nuclear model: 0 = point
        atm_rec[3] = ptr_zeta            # Pointer to zeta in _env
        _atm.append(atm_rec)

    return np.array(_atm, dtype=np.int32)

def make_bas(self):
    """Construct _bas array from Mole object. Returns _bas array and updated _env."""
    _bas = []
    env = list(self._env)  # Start with existing _env
    ptr = len(env)
    exp_idx = 0

    for i, atom1 in enumerate(self.iatsh):
        atom_id = atom1 - 1  # zero-based
        l_raw = self.mssh[i]
        nprim = self.mnsh[i]
        exponents = self.expsh[exp_idx:exp_idx + nprim]
        coeff1 = self.c1[exp_idx:exp_idx + nprim]
        coeff2 = self.c2[exp_idx:exp_idx + nprim] if self.c2 is not None else None
        exp_idx += nprim

        if l_raw == -1:
            # SP shell: add two basis functions (S and P parts)
            env.extend(exponents)
            env.extend(coeff1)
            env.extend(coeff2)
            # S shell part
            _bas.append([
                atom_id, 0, nprim, 1, int(self.cart),  # l=0
                ptr, ptr + nprim, 0
            ])
            # P shell part
            _bas.append([
                atom_id, 1, nprim, 0, int(self.cart),  # l=1
                ptr, ptr + 2*nprim, 0
            ])
            ptr += 3 * nprim
        else:
            l = abs(l_raw)
            env.extend(exponents)
            env.extend(coeff1)
            _bas.append([
                atom_id, l, nprim, 1, 0,
                ptr, ptr + nprim, 0
            ])
            ptr += 2 * nprim  # Exps + Coeffs

    # Update self._env and return _bas
    return np.array(_bas, dtype=np.int32)


def make_basis(mol):
    """
    Constructs a PySCF-style basis dict (mol._basis) from Mole object.
    Returns a dictionary like {'H': [[0, [exp, coeff], ...], ...], ...}
    """
    from pyscf.gto import gto_norm
    from pyscf.gto.mole import _nomalize_contracted_ao
    basis_dict = {}
    dones = []
    idx = 0  # current position in the primitive arrays
    processed = {}

    for shell_idx, atom_idx in enumerate(mol.iatsh):
        atom_symbol = mol.atomic_symbols[atom_idx - 1]
        if atom_symbol in dones and processed[atom_symbol] != atom_idx:
            nprim = mol.mnsh[shell_idx]
            idx += nprim
            continue

        if atom_symbol not in dones:
            dones.append(atom_symbol)
            processed[atom_symbol] = atom_idx

        shell_type = mol.mssh[shell_idx]
        nprim = mol.mnsh[shell_idx]

        exps = np.array(mol.expsh[idx:idx + nprim])
        coeffs1 = np.array(mol.c1[idx:idx + nprim])
        coeffs2 = np.array(mol.c2[idx:idx + nprim]) if mol.c2 is not None else None
        idx += nprim

        shell_entry = []

        if shell_type == -1:
            for e, c1_val, c2_val in zip(exps, coeffs1, coeffs2):
                shell.append([e, c1_val, c2_val])
            shell_entry = [shell]
        else:
            l = abs(shell_type)
            cs1 = coeffs1
            #cs1 = (gto_norm(l, exps) * coeffs1).reshape(-1, 1)
            #cs1 = _nomalize_contracted_ao(l, exps, cs1)
            shell = [abs(shell_type)]
            for e, c in zip(exps, cs1):
                shell.append([e, c])
            shell_entry = [shell]

        if atom_symbol not in basis_dict:
            basis_dict[atom_symbol] = shell_entry
        else:
            basis_dict[atom_symbol].extend(shell_entry)

    return basis_dict


def degens(l, cart_flag):
    """Return the number of contracted functions for a given angular momentum l.

    Args:
        l (int): Angular momentum quantum number.
        cart_flag (bool): If True, assume Cartesian functions; otherwise pure spherical.

    Returns:
        int: Number of contracted functions.
    """
    if cart_flag:
        return (l + 1) * (l + 2) // 2  # Cartesian: 1 (s), 3 (p), 6 (d), etc.
    else:
        return 2 * l + 1  # Spherical: 1 (s), 3 (p), 5 (d), etc.



class Mole:
    def __init__(self, path):
        self.path = path
        self.basis = read_level_theory(self.path)[-1]
        self.nalpha = int(read_from_fchk('Number of alpha electrons', self.path)[-1])
        self.nbeta = int(read_from_fchk('Number of beta electrons', self.path)[-1])
        self.charge = int(read_from_fchk('Charge', self.path)[-1])
        self.spin = int(self.nalpha) - int(self.nbeta)
        self.nelec = int(read_from_fchk('Number of electrons', self.path)[-1])
        self.charge = int(read_from_fchk('Charge', self.path)[-1])
        self.natoms = int(read_from_fchk('Number of atoms', self.path)[-1])
        self.natm = self.natoms
        self.nbasis = int(read_from_fchk('Number of basis functions', self.path)[-1])
        self.nao = self.nbasis
        self.atomic_numbers = [int(i) for i in read_list_from_fchk('Atomic numbers', 'Nuclear charges', self.path)]
        self.atomic_symbols = read_atomic_symbols(self.atomic_numbers)
        self.dcart = read_from_fchk("Pure/Cartesian d shells", self.path)[-1]
        self.fcart = read_from_fchk("Pure/Cartesian f shells", self.path)[-1]
        self.cart = self.dcart == 0 or self.fcart == 0
        self.verbose = 0
        self.mf = MeanField(path, self)
        self.iatsh = self.mf.iatsh
        self.mssh = self.mf.mssh
        self.mnsh = self.mf.mnsh
        self.expsh = self.mf.expsh
        self.c1 = self.mf.c1
        self.c2 = self.mf.c2 if hasattr(self.mf, 'c2') and self.mf.c2 is not None else None
        self.ncshell = self.mf.ncshell

        from pyscf import gto

        self._basis = make_basis(self)

        pyscf_mol = gto.M(
            atom=[(self.atomic_symbols[i], self.atom_coords()[i]) for i in range(self.natoms)],
            basis="cc-pvdz",
            spin=self.spin,
            charge=self.charge,
            cart=self.cart,
        )
        pyscf_mol.build()
        print(pyscf_mol.intor_symmetric("int1e_ovlp"))
        exit()

        self.copy = pyscf_mol
        self._atm = pyscf_mol._atm
        self._env = pyscf_mol._env
        self._bas = pyscf_mol._bas


        #self._bas = build_bas(self)
        #self._bas = pyscf_mol._bas
        #self._basis = pyscf_mol._basis
        #self._atm, self._env = make_atm_env(self.atomic_numbers, self.atom_coords())
        #self._env = pyscf_mol._env
        self._add_suffix = pyscf_mol._add_suffix
        self.nbas = len(self._bas)

    def atom_symbol(self, pos):
        return self.atomic_symbols[pos]

    def aoslice_by_atom(self, ao_loc=None):
        return self.copy.aoslice_by_atom()

    def atom_coords(self):
        coords = read_list_from_fchk('Current cartesian coordinates', 'Number of symbols', self.path)
        return np.array(coords).reshape(int(self.natoms), 3)

    def intor_symmetric(self,str):
        if str == 'int1e_ovlp':
            return self.copy.intor_symmetric('int1e_ovlp', comp=1)
        else:
            raise ValueError("Invalid integral type: {}".format(str))

    def atom_nelec_core(self, atm_id):
        '''Number of core electrons for pseudo potential.'''
        nuclear_charges = read_list_from_fchk('Nuclear charges', 'Current cartesian coordinates', self.path)
        core_electrons = [self.atomic_numbers[i] - int(nuclear_charges[i]) for i in range(self.natoms)]
        return core_electrons[atm_id]

    def ao_loc_nr(self):
        return self.copy.ao_loc_nr()

    def nao_nr(self):
        '''Number of basis functions.
        '''
        if self.cart:
            l = self._bas[:, 1]
            return int(((l + 1) * (l + 2) // 2 * self._bas[:, 3]).sum())
        else:
            return int(((self._bas[:,1]*2+1) * self._bas[:,3]).sum())

    def bas_atom(self, atm_id):
        '''Atom id for each basis function.
        '''
        return int(self._bas[atm_id, 0])

    def bas_angular(self, atm_id):
        '''Angular momentum for each basis function.
        '''
        return int(self._bas[atm_id, 1])

    def bas_nctr(self, atm_id):
        '''Number of contracted functions for each basis function.
        '''
        return int(self._bas[atm_id, 3])

    def atom_pure_symbol(self, atm_id):
        '''Atom symbol for each basis function.
        '''
        return self.copy.atom_pure_symbol(atm_id)

    def has_ecp(self):
        '''Check if the molecule has ECP.
        '''
        nuclear_charges = read_list_from_fchk('Nuclear charges', 'Current cartesian coordinates', self.path)
        nuclear_charges = [int(charge) for charge in nuclear_charges]
        return nuclear_charges != self.atomic_numbers

class MeanField:
    def __init__(self, path, mol):
        self.path = path
        self.mol = mol
        self.nalpha = self.mol.nalpha
        self.__class__.__name__ = read_level_theory(self.path)[-2]
        if 'R' in self.__class__.__name__:
            wf = 'rest'
            if 'HF' in self.__class__.__name__:
                self.__class__.__name__ = 'RHF'
            else:
                self.__class__.__name__ = 'RKS'
            self.orbital_energies = read_list_from_fchk('Alpha Orbital Energies', 'Alpha MO coefficients', self.path)
            nocc = 0
            for num in self.orbital_energies:
                if num > 0:
                    nocc += 1
            self.mo_occ = np.array([2.] * (int(self.mol.nalpha + self.mol.nbeta) // 2) + [0.] * (len(self.orbital_energies) - (int(self.mol.nalpha+self.mol.nbeta) // 2)))
        elif 'U' in self.__class__.__name__:
            wf = 'unrest'
            if 'HF' in self.__class__.__name__:
                self.__class__.__name__ = 'UHF'
            else:
                self.__class__.__name__ = 'UKS'
            self.alpha_orbital_energies = read_list_from_fchk('Alpha Orbital Energies', 'Beta Orbital Energies',
                                                              self.path)
            self.beta_orbital_energies = read_list_from_fchk('Beta Orbital Energies', 'Alpha MO coefficients',
                                                             self.path)
            nocc_alpha, nocc_beta = 0, 0
            for num in self.alpha_orbital_energies:
                if num > 0:
                    nocc_alpha += 1
            for num in self.beta_orbital_energies:
                if num > 0:
                    nocc_beta += 1
            self.mo_occ = [1] * nocc_alpha + [1] * nocc_beta + [0] * (
                        len(self.alpha_orbital_energies) + len(self.beta_orbital_energies) - nocc_alpha - nocc_beta)
        self.e_tot = float(read_from_fchk('SCF Energy', self.path)[-1])
        self.natoms = int(read_from_fchk('Number of atoms', self.path)[-1])
        self.nbasis = int(read_from_fchk('Number of basis functions', self.path)[-1])
        self.numprim = int(read_from_fchk('Number of primitive shells', self.path)[-1])
        self.atomic_numbers = [int(i) for i in read_list_from_fchk('Atomic numbers', 'Nuclear charges', self.path)]
        self.atomic_symbols = read_atomic_symbols(self.atomic_numbers)
        self.numao = int(read_from_fchk('Number of basis functions', self.path)[-1])
        self.nummo = int(read_from_fchk('Number of independent functions', self.path)[-1])
        # Number of contracted shells
        self.ncshell = int(read_from_fchk('Number of contracted shells', self.path)[-1])

        if wf == 'rest':
            self.mo_coeff = list(read_list_from_fchk('Alpha MO coefficients', 'Orthonormal basis', self.path))
            self.mo_coeff = np.array(self.mo_coeff).reshape(self.nummo, self.numao).T
        elif wf == 'unrest':
            self.mo_coeff_alpha = read_list_from_fchk('Alpha MO coefficients', 'Beta MO coefficients', self.path)
            self.mo_coeff_beta = read_list_from_fchk('Beta MO coefficients', 'Orthonormal basis', self.path)
            self.mo_coeff = [self.mo_coeff_alpha.T, self.mo_coeff_beta.T]
        # Number of primitive shells
        self.npshell = read_from_fchk('Number of primitive shells', self.path)[-1]
        # 0=s, 1=p, -1=sp, 2=6d, -2=5d, 3=10f, -3=7f
        # Shell types
        self.mssh = read_list_from_fchk('Shell types', 'Number of primitives per shell', self.path)
        self.mssh = [int(i) for i in self.mssh]
        # Number of primitives per shell
        self.mnsh = read_list_from_fchk('Number of primitives per shell', 'Shell to atom map', self.path)
        self.mnsh = [int(i) for i in self.mnsh]
        # Shell to atom map
        self.iatsh = read_list_from_fchk('Shell to atom map', 'Primitive exponents', self.path)
        self.iatsh = [int(i) for i in self.iatsh]
        # Primitive exponents
        self.expsh = read_list_from_fchk('Primitive exponents', 'Contraction coefficients', self.path)
        # Current cartesian coordinates
        coords = read_list_from_fchk('Current cartesian coordinates', 'Number of symbols', self.path)
        self.coord = np.array(coords).reshape(int(self.natoms), 3)

        with open(path, 'r') as file:
            if 'P(S=P)' in file.read():
                self.c1 = read_list_from_fchk('Contraction coefficients', 'P(S=P) Contraction coefficients', self.path)
                self.c2 = read_list_from_fchk('P(S=P) Contraction coefficients', 'Coordinates of each shell', self.path)
            else:
                self.c1 = read_list_from_fchk('Contraction coefficients', 'Coordinates of each shell', self.path)
                self.c2 = None

    def make_rdm1(self):
        """Computes density matrix."""
        if len(np.shape(self.mo_coeff)) == 2:  # RHF/RKS
            # Ensure mo_occ is correctly sized and used for slicing
            occupied_indices = np.where(self.mo_occ > 0)[0]
            mo_occ_coeffs = self.mo_coeff[:, occupied_indices]
            return np.dot(mo_occ_coeffs, mo_occ_coeffs.T) * 2
        elif len(np.shape(self.mo_coeff)) == 3:  # UHF/UKS
            # Temporarily closed, come back later :)
            return None
            occupied_indices_a = np.where(self.mo_occ[0] > 0)[0]
            occupied_indices_b = np.where(self.mo_occ[1] > 0)[0]
            mo_a_occ = self.mo_coeff[0][:, occupied_indices_a]
            mo_b_occ = self.mo_coeff[1][:, occupied_indices_b]
            rdm1_a = np.dot(mo_a_occ, mo_a_occ.T)
            rdm1_b = np.dot(mo_b_occ, mo_b_occ.T)
            return np.array([rdm1_a, rdm1_b])

    def get_ovlp(self):
        return self.mol.copy.intor_symmetric('int1e_ovlp')
        process_basis(self)
        ovlp = build_ovlp(self)
        return ovlp

def mult(shell_type):
    mult_dict = { #s=1, p=2, d=3, f=4, g=5, sp=-1
         0: 1,  1: 3,  2: 6,  3: 10, 4: 15, 5: 21,
         - 5: 11, -4: 9, -3: 7, -2: 5, -1: 4,
    }
    return mult_dict.get(shell_type, 0)

def process_basis(mf):

    numprim, nbasis = 0, 0
    for i in range(mf.ncshell):
        if mf.mssh[i] < -1:
            numprim += mult(abs(mf.mssh[i])) * mf.mnsh[i]
        else:
            numprim += mult(mf.mssh[i]) * mf.mnsh[i]
        nbasis += mult(mf.mssh[i])

    # Basis to atom map
    ihold = [0] * nbasis
    ii = 0
    for i in range(mf.ncshell):
        for k in range(mult(mf.mssh[i])):
            ii += 1
            ihold[ii - 1] = mf.iatsh[i]

    # Setting basis set limits for Mulliken
    llim = [1]
    iulim = [0] * mf.natoms
    iat = 1

    for i in range(nbasis):
        if ihold[i] == iat:
            if iat - 1 < len(iulim):
                iulim[iat - 1] = i
            llim.append(i + 1)
            iat += 1
    if iat - 1 < len(iulim):
        iulim[iat - 1] = nbasis

    icount, jcount = 0, 0
    expp, coefp = [0] * numprim, [0] * numprim

    for i in range(mf.ncshell):
        for j in range(mf.mnsh[i]):
            kk = mult(abs(mf.mssh[i]))
            if mf.mssh[i] == -1:
                kk = 4
            for k in range(kk):
                expp[icount] = mf.expsh[jcount]
                if mf.mssh[i] == -1 and k != 0:
                    coefp[icount] = mf.c2[jcount]
                else:
                    coefp[icount] = mf.c1[jcount]
                icount += 1
            jcount += 1

    nlm, iptoat = get_nlm(mf)

    xnorm = np.zeros(numprim)

    for i in range(numprim):
        nn = nlm[i, 0]
        ll = nlm[i, 1]
        mm = nlm[i, 2]
        fnn = factorial(nn) / factorial(2 * nn)
        fll = factorial(ll) / factorial(2 * ll)
        fmm = factorial(mm) / factorial(2 * mm)
        xnorm[i] = (2.0 * expp[i] / np.pi) ** 0.75 * np.sqrt((8.0 * expp[i]) ** (nn + ll + mm) * fnn * fll * fmm)

    mf.coefp = coefp
    mf.xnorm = xnorm
    mf.numprim = numprim
    mf.nbasis = nbasis
    coefpb, iptob_cartesian = prim_orb_mapping(mf)

    mmax = 0
    for i in range(mf.ncshell):
        ii = mf.mnsh[i]
        if mf.mssh[i] <= -2:
            ii = 3 * ii
        if mf.mssh[i] == -4:
            ii = 2 * ii
        if ii > mmax:
            mmax = ii
    mmax += 1

    nprimbas = np.zeros((mmax, nbasis), dtype=int)

    for i in range(nbasis):
        npb = 0
        for j in range(numprim):
            if abs(coefpb[j, i]) > 1.0e-10:
                nprimbas[npb, i] = j + 1
                npb += 1

    # Updating the mf object
    mf.numprim = numprim
    mf.nbasis = nbasis
    mf.ihold = ihold
    mf.llim = llim
    mf.iulim = iulim
    mf.nlm = nlm
    mf.iptoat = iptoat
    mf.xnorm = xnorm
    mf.coefpb = coefpb
    mf.iptob_cartesian = iptob_cartesian
    mf.mmax = mmax
    mf.nprimbas = nprimbas
    mf.expp = expp

def build_ovlp(mf):
    """
    Calculates the overlap matrix using partial NumPy vectorization.
    This is the most efficient pure Python/NumPy approach without Numba/PySCF.

    Args:
        mf: An object containing molecular information:
            numprim (int): Number of primitive Gaussian functions.
            nbasis (int): Number of basis functions.
            coord (array-like): Atomic coordinates, shape (num_atoms, 3).
            iptoat (array-like): Mapping from primitive index (1-based) to atom index.
            nlm (array-like): Angular momentum numbers (lx, ly, lz) for each primitive, shape (numprim, 3).
            expp (array-like): Exponents for each primitive, shape (numprim,).
            coefpb (array-like): Contraction coefficients, ASSUMED shape (numprim, nbasis).
            # nprimbas: Not used if coefpb is the C matrix.

    Returns:
        np.ndarray: The overlap matrix S, shape (nbasis, nbasis).
    """
    tol = 1.0e-8
    numprim = mf.numprim
    nbasis = mf.nbasis

    coord = np.asarray(mf.coord)
    iptoat = np.asarray(mf.iptoat) - 1 # 0-based
    nlm = np.asarray(mf.nlm)
    expp = np.asarray(mf.expp)
    coefpb = np.asarray(mf.coefpb)

    # --- Part 1: Calculate Primitive Overlaps (sp) ---

    # 1. Vectorized calculation of pairwise distances and exponents
    coords_p = coord[iptoat]
    AminusB_xyz = coords_p[:, None, :] - coords_p[None, :, :] # Shape (numprim, numprim, 3)
    AminusB2_xyz = AminusB_xyz**2

    expp_ia = expp[:, None]
    expp_ib = expp[None, :]
    gamma_p = expp_ia + expp_ib
    # Add epsilon if concerned about division by zero, though unlikely for positive exponents
    # gamma_p = gamma_p + 1e-30
    eta_p = (expp_ia * expp_ib) / gamma_p

    sp_base = (np.pi / gamma_p)**(1.5) # Base overlap factor

    # 2. Loop for the complex angular momentum part (Python loops remain here)
    sp = np.zeros((numprim, numprim))
    idx_ia, idx_ib = np.triu_indices(numprim) # Efficiently get upper triangle indices

    for ia, ib in zip(idx_ia, idx_ib):
        # Extract pre-calculated values for this pair
        AmB = AminusB_xyz[ia, ib]
        AmB2 = AminusB2_xyz[ia, ib]
        etap_val = eta_p[ia, ib]
        exp_ia_val = expp[ia]
        exp_ib_val = expp[ib]
        nlm_ia = nlm[ia]
        nlm_ib = nlm[ib]

        # --- Check for zero overlap (Original logic) ---
        skip = False
        for ixyz in range(3):
            if abs(AmB[ixyz]) < tol and (nlm_ia[ixyz] + nlm_ib[ixyz]) % 2 != 0:
                skip = True
                break
        if skip:
            # sp[ia, ib] remains 0.0
            continue
        # --- End Check ---

        # --- Calculate angular factor product (Innermost loops) ---
        angular_factor_product = 1.0
        for ixyz in range(3): # Loop x, y, z
            l1 = nlm_ia[ixyz]
            l2 = nlm_ib[ixyz]
            AmB_ax = AmB[ixyz]

            # Use scipy's factorial (or your own if scipy not available)
            term_xyz = factorial(l1) * factorial(l2) / (2.0**(l1 + l2))

            if abs(AmB_ax) > tol:
                term_xyz *= np.exp(-etap_val * AmB2[ixyz])

            # --- sum_i calculation (Original loop structure) ---
            sum_i = 0.0
            for i1 in range(l1 // 2 + 1):
                j1 = l1 - 2 * i1
                for i2 in range(l2 // 2 + 1):
                    j2 = l2 - 2 * i2
                    j = j1 + j2

                    # --- sum_r calculation (Original loop structure & logic) ---
                    sum_r = 0.0
                    if abs(AmB_ax) > tol:
                        for ir in range(j // 2 + 1):
                            fac_ir = factorial(ir)
                            fac_j_2ir = factorial(j - 2 * ir)
                            if fac_ir == 0 or fac_j_2ir == 0: continue # Avoid division by zero
                            den = fac_ir * fac_j_2ir

                            num = (2.0 * AmB_ax)**float(j - 2 * ir)
                            xfac = etap_val**float(j - ir) * num / den

                            if ir % 2 != 0: xfac = -xfac
                            sum_r += xfac
                    elif j % 2 == 0:
                        k_idx = j // 2
                        fac_k = factorial(k_idx)
                        if fac_k == 0: continue
                        sum_r = etap_val**float(k_idx) / fac_k
                        if k_idx % 2 != 0: sum_r = -sum_r
                    # --- End sum_r ---

                    if j1 % 2 != 0: sum_r_signed = -sum_r
                    else: sum_r_signed = sum_r

                    fac_i1 = factorial(i1); fac_j1 = factorial(j1)
                    fac_i2 = factorial(i2); fac_j2 = factorial(j2)
                    fac_j = factorial(j)
                    facij_denom_part = (fac_i1 * fac_j1 * fac_i2 * fac_j2)
                    exp_pow_part = exp_ia_val**float(l1 - i1) * exp_ib_val**float(l2 - i2)
                    facij_orig = facij_denom_part * exp_pow_part

                    if facij_orig != 0:
                        sum_i += sum_r_signed * float(fac_j) / facij_orig
            # --- End sum_i ---
            angular_factor_product *= sum_i * term_xyz
        # --- End angular factor product ---

        # Assign value to sp matrix (upper triangle)
        sp[ia, ib] = sp_base[ia, ib] * angular_factor_product

    # 3. Fill lower triangle for symmetry
    sp = sp + sp.T - np.diag(np.diag(sp))


    # --- Part 2: Vectorized Contraction ---
    # Replace the slow nested 'while' loops with matrix multiplication
    # Assumes coefpb IS the contraction matrix C (numprim, nbasis)
    # s = C^T @ sp @ C
    s = coefpb.T @ sp @ coefpb

    return s

def build_ovlp2(mf):
    tol = 1.0e-8
    numprim = mf.numprim
    nbasis = mf.nbasis
    coord = np.array(mf.coord)
    iptoat = np.array(mf.iptoat)
    nlm = mf.nlm
    expp = mf.expp
    coefpb = mf.coefpb
    nprimbas = mf.nprimbas

    sp = np.zeros((numprim, numprim))
    s = np.zeros((nbasis, nbasis))

    for ia in range(numprim):
        for ib in range(ia+1):
            sp[ia, ib] = 0.0
            AminusB = coord[iptoat[ia]-1] - coord[iptoat[ib]-1]
            for ixyz in range(3):
                ii = (nlm[ia, ixyz] + nlm[ib, ixyz]) % 2
                if abs(AminusB[ixyz]) < tol and ii != 0:
                    break
            gamma_p = expp[ia] + expp[ib]
            eta_p = (expp[ia] * expp[ib]) / gamma_p
            do_ov = np.pi ** (3.0 / 2.0) / gamma_p ** (3.0 / 2.0)
            for ixyz in range(3):
                do_ov *= factorial(nlm[ia, ixyz]) * factorial(nlm[ib, ixyz]) / (2.0 ** (nlm[ia, ixyz] + nlm[ib, ixyz]))
                if abs(AminusB[ixyz]) > tol:
                    do_ov *= np.exp(-eta_p * AminusB[ixyz] ** 2.0)
                sum_i = 0.0
                for i1 in range(nlm[ia, ixyz] // 2 + 1):
                    j1 = nlm[ia, ixyz] - 2 * i1
                    for i2 in range(nlm[ib, ixyz] // 2 + 1):
                        j2 = nlm[ib, ixyz] - 2 * i2
                        j = j1 + j2
                        facij = factorial(i1) * factorial(j1) * factorial(i2) * factorial(j2) * expp[ia] ** (nlm[ia, ixyz] - i1) * expp[ib] ** (nlm[ib, ixyz] - i2)
                        sum_r = 0.0
                        if abs(AminusB[ixyz]) > tol:
                            for ir in range(j // 2 + 1):
                                xfac = eta_p ** (j - ir) * (2.0 * AminusB[ixyz]) ** (j - 2 * ir) / (factorial(ir) * factorial(j - 2 * ir))
                                if ir % 2 != 0:
                                    xfac = -xfac
                                sum_r += xfac
                        elif j % 2 == 0:
                            sum_r = eta_p ** (j // 2) / factorial(j // 2)
                            if (j // 2) % 2 != 0:
                                sum_r = -sum_r
                        if j1 % 2 != 0:
                            sum_r = -sum_r
                        sum_i += sum_r * factorial(j) / facij
                do_ov *= sum_i
            sp[ia, ib] = do_ov
            if ia != ib:
                sp[ib, ia] = sp[ia, ib]

    for i in range(nbasis):
        for j in range(i + 1):
            s[i, j] = 0.0
            k = 0
            while nprimbas[k, i] != 0:
                l = 0
                while nprimbas[l, j] != 0:
                    s[i, j] += coefpb[nprimbas[k, i]-1, i] * coefpb[nprimbas[l, j]-1, j] * sp[nprimbas[k, i]-1, nprimbas[l, j]-1]
                    l += 1
                k += 1
            if i != j:
                s[j, i] = s[i, j]

    return np.array(s)


def get_nlm(mf):
    ncshell = mf.ncshell
    mssh = mf.mssh
    mnsh = mf.mnsh
    iatsh = mf.iatsh

    nlm = []
    iptoat = []
    icount = 0

    for i in range(ncshell):
        if mssh[i] == 0:  # S-type orbitals
            for j in range(mnsh[i]):
                iptoat.append(iatsh[i])
                nlm.append([0, 0, 0])
                icount += 1
        elif mssh[i] == 1:  # P-type orbitals
            for j in range(mnsh[i]):
                nlm.extend([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
                iptoat.extend([iatsh[i]] * 3)
                icount += 3
        elif mssh[i] == -1:  # SP-type orbitals
            for j in range(mnsh[i]):
                nlm.extend([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
                iptoat.extend([iatsh[i]] * 4)
                icount += 4
        elif abs(mssh[i]) == 2:  # D-type orbitals
            for j in range(mnsh[i]):
                nlm.extend([[2, 0, 0], [0, 2, 0], [0, 0, 2], [1, 1, 0], [1, 0, 1], [0, 1, 1]])
                iptoat.extend([iatsh[i]] * 6)
                icount += 6
        elif abs(mssh[i]) == 3:  # F-type orbitals
            for j in range(mnsh[i]):
                nlm.extend([
                    [3, 0, 0], [0, 3, 0], [0, 0, 3], [1, 2, 0], [2, 1, 0], [2, 0, 1],
                    [1, 0, 2], [0, 1, 2], [0, 2, 1], [1, 1, 1]
                ])
                iptoat.extend([iatsh[i]] * 10)
                icount += 10
        elif abs(mssh[i]) == 4:  # G-type orbitals
            for j in range(mnsh[i]):
                nlm.extend([
                    [0, 0, 4], [0, 1, 3], [0, 2, 2], [0, 3, 1], [0, 4, 0], [1, 0, 3],
                    [1, 1, 2], [1, 2, 1], [1, 3, 0], [2, 0, 2], [2, 1, 1], [2, 2, 0],
                    [3, 0, 1], [3, 1, 0], [4, 0, 0]
                ])
                iptoat.extend([iatsh[i]] * 15)
                icount += 15
        else:
            raise ValueError('Angular momentum not implemented')

    return np.array(nlm), np.array(iptoat)

def prim_orb_mapping(mf):
    numprim = 0
    nbasis = 0
    ncshell = mf.ncshell
    mssh = mf.mssh
    mnsh = mf.mnsh
    coefp = mf.coefp
    xnorm = mf.xnorm
    coefpb = np.zeros((mf.numprim, mf.nbasis))  # Adjust the size as needed
    iptob_cartesian = np.zeros(len(coefp), dtype=int)

    for i in range(ncshell):
        for j in range(mnsh[i]):
            if mssh[i] >= -1:
                for k in range(mult(mssh[i])):
                    numprim += 1
                    coefpb[numprim - 1, nbasis+k] = coefp[numprim - 1] * xnorm[numprim - 1]
                    iptob_cartesian[numprim-1] = nbasis + k
            elif mssh[i] == -2:  # Mapping for pure 5d
                coefpb[numprim, nbasis] = coefp[numprim] * xnorm[numprim] * (-0.5)
                coefpb[numprim, nbasis+3] = coefp[numprim] * xnorm[numprim] * (np.sqrt(3.0) / 2.0)
                coefpb[numprim+1, nbasis] = coefp[numprim+1] * xnorm[numprim+1] * (-0.5)
                coefpb[numprim+1, nbasis+3] = coefp[numprim+1] * xnorm[numprim+1] * (-np.sqrt(3.0) / 2.0)
                coefpb[numprim+2, nbasis] = coefp[numprim+2] * xnorm[numprim+2]
                coefpb[numprim+3, nbasis+4] = coefp[numprim+3] * xnorm[numprim+3]
                coefpb[numprim+4, nbasis+1] = coefp[numprim+4] * xnorm[numprim+4]
                coefpb[numprim+5, nbasis+2] = coefp[numprim+5] * xnorm[numprim+5]
                numprim += mult(abs(mssh[i]))
            elif mssh[i] == -3:  # Mapping for pure 7f
                coefpb[numprim, nbasis+1] = coefp[numprim] * xnorm[numprim] * (-np.sqrt(6.0) / 4.0)
                coefpb[numprim, nbasis+5] = coefp[numprim] * xnorm[numprim] * (np.sqrt(10.0) / 4.0)
                coefpb[numprim+1, nbasis+2] = coefp[numprim+1] * xnorm[numprim+1] * (-np.sqrt(6.0) / 4.0)
                coefpb[numprim+1, nbasis+6] = coefp[numprim+1] * xnorm[numprim+1] * (-np.sqrt(10.0) / 4.0)
                coefpb[numprim+2, nbasis] = coefp[numprim+2] * xnorm[numprim+2]
                coefpb[numprim+3, nbasis+1] = coefp[numprim+3] * xnorm[numprim+3] * (-np.sqrt(30.0) / 20.0)
                coefpb[numprim+3, nbasis+5] = coefp[numprim+3] * xnorm[numprim+3] * (-3.0 * np.sqrt(2.0) / 4.0)
                coefpb[numprim+4, nbasis+2] = coefp[numprim+4] * xnorm[numprim+4] * (-np.sqrt(30.0) / 20.0)
                coefpb[numprim+4, nbasis+6] = coefp[numprim+4] * xnorm[numprim+4] * (3.0 * np.sqrt(2.0) / 4.0)
                coefpb[numprim+5, nbasis] = coefp[numprim+5] * xnorm[numprim+5] * (-3.0 * np.sqrt(5.0) / 10.0)
                coefpb[numprim+5, nbasis+3] = coefp[numprim+5] * xnorm[numprim+5] * (np.sqrt(3.0) / 2.0)
                coefpb[numprim+6, nbasis+1] = coefp[numprim+6] * xnorm[numprim+6] * (np.sqrt(30.0) / 5.0)
                coefpb[numprim+7, nbasis+2] = coefp[numprim+7] * xnorm[numprim+7] * (np.sqrt(30.0) / 5.0)
                coefpb[numprim+8, nbasis] = coefp[numprim+8] * xnorm[numprim+8] * (-3.0 * np.sqrt(5.0) / 10.0)
                coefpb[numprim+8, nbasis+3] = coefp[numprim+8] * xnorm[numprim+8] * (-np.sqrt(3.0) / 2.0)
                coefpb[numprim+9, nbasis+4] = coefp[numprim+9] * xnorm[numprim+9]
                numprim += mult(abs(mssh[i]))
            elif mssh[i] == -4:  # Mapping for pure 9f
                coefpb[numprim, nbasis] = coefp[numprim] * xnorm[numprim]
                coefpb[numprim+1, nbasis+2] = coefp[numprim+1] * xnorm[numprim+1] * (np.sqrt(70.0) / 7.0)
                coefpb[numprim+2, nbasis] = coefp[numprim+2] * xnorm[numprim+2] * (-3.0 * np.sqrt(105.0) / 35.0)
                coefpb[numprim+2, nbasis+3] = coefp[numprim+2] * xnorm[numprim+2] * (-3.0 * np.sqrt(21.0) / 14.0)
                coefpb[numprim+3, nbasis+2] = coefp[numprim+3] * xnorm[numprim+3] * (-3.0 * np.sqrt(70.0) / 28.0)
                coefpb[numprim+3, nbasis+6] = coefp[numprim+3] * xnorm[numprim+3] * (-np.sqrt(10.0) / 4.0)
                coefpb[numprim+4, nbasis] = coefp[numprim+4] * xnorm[numprim+4] * (3.0 / 8.0)
                coefpb[numprim+4, nbasis+3] = coefp[numprim+4] * xnorm[numprim+4] * (np.sqrt(5.0) / 4.0)
                coefpb[numprim+4, nbasis+7] = coefp[numprim+4] * xnorm[numprim+4] * (np.sqrt(35.0) / 8.0)
                coefpb[numprim+5, nbasis+1] = coefp[numprim+5] * xnorm[numprim+5] * (np.sqrt(70.0) / 7.0)
                coefpb[numprim+6, nbasis+4] = coefp[numprim+6] * xnorm[numprim+6] * (3.0 * np.sqrt(7.0) / 7.0)
                coefpb[numprim+7, nbasis+1] = coefp[numprim+7] * xnorm[numprim+7] * (-3.0 * np.sqrt(14.0) / 28.0)
                coefpb[numprim+7, nbasis+5] = coefp[numprim+7] * xnorm[numprim+7] * (-3.0 * np.sqrt(2.0) / 4.0)
                coefpb[numprim+8, nbasis+4] = coefp[numprim+8] * xnorm[numprim+8] * (-np.sqrt(35.0) / 14.0)
                coefpb[numprim+8, nbasis+8] = coefp[numprim+8] * xnorm[numprim+8] * (-np.sqrt(5.0) / 2.0)
                coefpb[numprim+9, nbasis] = coefp[numprim+9] * xnorm[numprim+9] * (-3.0 * np.sqrt(105.0) / 35.0)
                coefpb[numprim+9, nbasis+3] = coefp[numprim+9] * xnorm[numprim+9] * (3.0 * np.sqrt(21.0) / 14.0)
                coefpb[numprim+10, nbasis+2] = coefp[numprim+10] * xnorm[numprim+10] * (-3.0 * np.sqrt(14.0) / 28.0)
                coefpb[numprim+10, nbasis+6] = coefp[numprim+10] * xnorm[numprim+10] * (3.0 * np.sqrt(2.0) / 4.0)
                coefpb[numprim+11, nbasis] = coefp[numprim+11] * xnorm[numprim+11] * (3.0 * np.sqrt(105.0) / 140.0)
                coefpb[numprim+11, nbasis+7] = coefp[numprim+11] * xnorm[numprim+11] * (-3.0 * np.sqrt(3.0) / 4.0)
                coefpb[numprim+12, nbasis+1] = coefp[numprim+12] * xnorm[numprim+12] * (-3.0 * np.sqrt(70.0) / 28.0)
                coefpb[numprim+12, nbasis+5] = coefp[numprim+12] * xnorm[numprim+12] * (np.sqrt(10.0) / 4.0)
                coefpb[numprim+13, nbasis+4] = coefp[numprim+13] * xnorm[numprim+13] * (-np.sqrt(35.0) / 14.0)
                coefpb[numprim+13, nbasis+8] = coefp[numprim+13] * xnorm[numprim+13] * (np.sqrt(5.0) / 2.0)
                coefpb[numprim+14, nbasis] = coefp[numprim+14] * xnorm[numprim+14] * (3.0 / 8.0)
                coefpb[numprim+14, nbasis+3] = coefp[numprim+14] * xnorm[numprim+14] * (-np.sqrt(5.0) / 4.0)
                coefpb[numprim+14, nbasis+7] = coefp[numprim+14] * xnorm[numprim+14] * (np.sqrt(35.0) / 8.0)
                numprim += mult(abs(mssh[i]))
        nbasis += mult(mssh[i])

    return coefpb, iptob_cartesian
