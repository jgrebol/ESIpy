import numpy as np
from math import factorial


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

def is_cart(path):
    with open(path, 'r') as file:
        lines = file.readlines()
        if len(lines) >= 10:
            return '6d' in lines[9]
        else:
            return False

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

class Mole:
    def __init__(self, path):
        self.path = path
        self.basis = read_level_theory(self.path)[-1]
        self.nalpha = int(read_from_fchk('Number of alpha electrons', self.path)[-1])
        self.nbeta = int(read_from_fchk('Number of beta electrons', self.path)[-1])
        self.charge = int(read_from_fchk('Charge', self.path)[-1])
        self.spin = int(self.nalpha) - int(self.nbeta)
        self.nelec = int(read_from_fchk('Number of electrons', self.path)[-1])
        self.natoms = int(read_from_fchk('Number of atoms', self.path)[-1])
        self.charge = int(read_from_fchk('Charge', self.path)[-1])
        self.natm = self.natoms
        self.nbasis = int(read_from_fchk('Number of basis functions', self.path)[-1])
        self.nbas = self.nbasis
        self.nao = self.nbasis
        self.atomic_numbers = [int(i) for i in read_list_from_fchk('Atomic numbers', 'Nuclear charges', self.path)]
        self.atomic_symbols = read_atomic_symbols(self.atomic_numbers)
        self.iatsh = read_list_from_fchk('Shell to atom map', 'Primitive exponents', self.path)
        self.iatsh = [int(i) for i in self.iatsh]
        self.mssh = read_list_from_fchk('Shell types', 'Number of primitives per shell', self.path)
        self.mssh = [int(i) for i in self.mssh]
        self.dcart = read_from_fchk("Pure/Cartesian d shells", self.path)[-1]
        self.fcart = read_from_fchk("Pure/Cartesian f shells", self.path)[-1]
        self.cart = is_cart(path)

    def atom_symbol(self, pos):
        return self.atomic_symbols[pos]

    def aoslice_by_atom(self):

        counts = [0] * (max(self.iatsh) + 1)
        for value in self.iatsh:
            counts[value] += 1
        counts = counts[1:]

        start_shell = 0
        aoslices = []

        result = []
        index = 0
        for count in counts:
            sums = sum(mult(self.mssh[index + i]) for i in range(count))
            result.append(sums)
            index += count

        for s in result:
            stop_shell = start_shell + s
            aoslices.append(np.array((0, 0, start_shell, stop_shell)))
            start_shell = stop_shell

        return np.array(aoslices)

    def atom_coords(self):
        coords = read_list_from_fchk('Current cartesian coordinates', 'Number of symbols', self.path)
        return np.array(coords).reshape(int(self.natoms), 3)


class MeanField:
    def __init__(self, path, mol):
        self.path = path
        self.mol = mol
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
        # Number of contracted shells
        self.ncshell = int(read_from_fchk('Number of contracted shells', self.path)[-1])
        if wf == 'rest':
            self.mo_coeff = list(read_list_from_fchk('Alpha MO coefficients', 'Orthonormal basis', self.path))
            self.mo_coeff = np.array(self.mo_coeff).reshape(int(np.sqrt(len(self.mo_coeff))), -1).T
            self.ncshell = int(self.ncshell / 2)
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

    def get_ovlp(self):
        from pyscf import gto

        mol = self.mol

        mol_pyscf = gto.M(
            atom=[(mol.atomic_symbols[i], mol.atom_coords()[i]) for i in range(mol.natoms)],
            basis=mol.basis,
            spin=mol.spin,
            charge=mol.charge,
            cart=mol.cart,
        )
        mol_pyscf.build()

        ovlp = mol_pyscf.intor("int1e_ovlp")
        return np.array(ovlp)

def mult(shell_type):
    mult_dict = { #s=1, p=2, d=3, f=4, g=5, sp=-1
         0: 1,  1: 3,  2: 6,  3: 10, 4: 15, 5: 21,
         - 5: 11, -4: 9, -3: 7, -2: 5, -1: 4,
    }
    return mult_dict.get(shell_type, 0)

def process_basis(mf):
    numprim, nbasis = 0, 0
    for i in range(mf.ncshell):
        if mf.mssh[i] <= -1:
            numprim += mult(abs(mf.mssh[i])) * mf.mnsh[i]
        else:
            numprim += mult(mf.mssh[i]) * mf.mnsh[i]
        nbasis += mult(mf.mssh[i])

    # Basis to atom map
    ihold = []
    for i in range(mf.ncshell):
        for k in range(mult(mf.mssh[i])):
            ihold.append(mf.iatsh[i])

    # Setting basis set limits for Mulliken
    llim = [1]
    iulim = [0] * mf.natoms
    iat = 0

    for i in range(nbasis):
        if ihold[i] != iat:
            iulim[iat - 1] = i
            llim.append(i + 1)
            iat += 1
    iulim[iat - 1] = mf.nbasis

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
    numprim = numprim

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

def build_ovlp(mf):
    tol = 1.0e-8
    numprim = len(mf.expsh)
    nbasis = mf.nbasis
    coord = mf.coord
    iptoat = mf.iptoat
    nlm = mf.nlm
    expp = mf.expsh
    coefpb = mf.coefpb
    nprimbas = mf.nprimbas

    sp = np.zeros((numprim, numprim))
    s = np.zeros((nbasis, nbasis))

    for ia in range(numprim):
        for ib in range(ia):
            sp[ia, ib] = 0.0
            AminusB = coord[iptoat[ia]] - coord[iptoat[ib]]
            for ixyz in range(3):
                ii = (nlm[ia, ixyz] + nlm[ib, ixyz]) % 2
                if abs(AminusB[ixyz]) < tol and ii != 0:
                    break
            else:
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
    numprim = mf.numprim

    nlm = np.zeros((numprim, 3), dtype=int)
    iptoat = np.zeros(numprim, dtype=int)
    icount = 0

    for i in range(ncshell):
        if mssh[i] == 0:  # S-type orbitals
            for j in range(mnsh[i]):
                iptoat[icount] = iatsh[i]
                icount += 1
        elif mssh[i] == 1:  # P-type orbitals
            for j in range(mnsh[i]):
                nlm[icount:icount + 3] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
                iptoat[icount:icount + 3] = iatsh[i]
                icount += 3
        elif mssh[i] == -1:  # SP-type orbitals
            for j in range(mnsh[i]):
                nlm[icount:icount + 4] = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
                iptoat[icount:icount + 4] = iatsh[i]
                icount += 4
        elif abs(mssh[i]) == 2:  # D-type orbitals
            for j in range(mnsh[i]):
                nlm[icount:icount + 6] = [[2, 0, 0], [0, 2, 0], [0, 0, 2], [1, 1, 0], [1, 0, 1], [0, 1, 1]]
                iptoat[icount:icount + 6] = iatsh[i]
                icount += 6
        elif abs(mssh[i]) == 3:  # F-type orbitals
            for j in range(mnsh[i]):
                nlm[icount:icount + 10] = [
                    [3, 0, 0], [0, 3, 0], [0, 0, 3], [1, 2, 0], [2, 1, 0], [2, 0, 1],
                    [1, 0, 2], [0, 1, 2], [0, 2, 1], [1, 1, 1]
                ]
                iptoat[icount:icount + 10] = iatsh[i]
                icount += 10
        elif abs(mssh[i]) == 4:  # G-type orbitals
            for j in range(mnsh[i]):
                nlm[icount:icount + 15] = [
                    [0, 0, 4], [0, 1, 3], [0, 2, 2], [0, 3, 1], [0, 4, 0], [1, 0, 3],
                    [1, 1, 2], [1, 2, 1], [1, 3, 0], [2, 0, 2], [2, 1, 1], [2, 2, 0],
                    [3, 0, 1], [3, 1, 0], [4, 0, 0]
                ]
                iptoat[icount:icount + 15] = iatsh[i]
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
