import numpy as np
from scipy.special import factorial2

def readfchk(filename):
    """
    Reads the information from a fchk file.
    Args:
        filename: Name of the fchk file.

    Returns:
        Mole object and MeanField object.
    """
    mol = Mole(filename)
    mf = MeanField(filename)
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
    print(f'Reading from {start} to {end}')
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
        self.nalpha = read_from_fchk('Number of alpha electrons', self.path)[-1]
        self.nbeta = read_from_fchk('Number of beta electrons', self.path)[-1]
        self.spin = int(self.nalpha) - int(self.nbeta)
        self.nelec = read_from_fchk('Number of electrons', self.path)[-1]
        self.natm = read_from_fchk('Number of atoms', self.path)[-1]
        self.nbas = read_from_fchk('Number of basis functions', self.path)[-1]

    def atom_coords(self):
        coords = read_list_from_fchk('Current cartesian coordinates', 'Number of symbols', self.path)
        return np.array(coords).reshape(int(self.natm), 3)


class MeanField:
    def __init__(self, path):
        self.path = path
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
            print(nocc)
            self.mo_occ = [2.] * nocc + [0.] * (len(self.orbital_energies) - nocc)
            self.mo_occ = np.array(self.mo_occ)
            print(self.mo_occ)
            exit()
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
        self.e_tot = read_from_fchk('SCF Energy', self.path)[-1]
        self.natm = read_from_fchk('Number of atoms', self.path)[-1]
        self.nbas = read_from_fchk('Number of basis functions', self.path)[-1]
        self.atomic_numbers = [int(i) for i in read_list_from_fchk('Atomic numbers', 'Nuclear charges', self.path)]
        self.atomic_symbols = read_atomic_symbols(self.atomic_numbers)
        if wf == 'rest':
            self.mo_coeff = read_list_from_fchk('Alpha MO coefficients', 'Orthonormal basis', self.path)
        elif wf == 'unrest':
            self.mo_coeff_alpha = read_list_from_fchk('Alpha MO coefficients', 'Beta MO coefficients', self.path)
            self.mo_coeff_beta = read_list_from_fchk('Beta MO coefficients', 'Orthonormal basis', self.path)
            self.mo_coeff = [self.mo_coeff_alpha, self.mo_coeff_beta]
        self.n_cont_shells = read_from_fchk('Number of contracted shells', self.path)[-1]
        self.n_prim_shells = read_from_fchk('Number of primitive shells', self.path)[-1]
        # 0=s, 1=p, -1=sp, 2=6d, -2=5d, 3=10f, -3=7f
        self.shell_types = read_list_from_fchk('Shell types', 'Number of primitives per shell', self.path)
        self.n_prim_per_shell = read_list_from_fchk('Number of primitives per shell', 'Shell to atom map', self.path)
        self.shell_to_atom_map = read_list_from_fchk('Shell to atom map', 'Primitive exponents', self.path)
        self.primitive_exps = read_list_from_fchk('Primitive exponents', 'Contraction coefficients', self.path)

        # with open('bz.fchk', 'r') as file:
        # if 'P(S=P)' in file.read():
        # self.contract_coeff = read_list_from_fchk('Contraction coefficients', 'P(S=P) Contraction coefficients', self.path)
        # self.psp_contract_coeff = read_list_from_fchk('P(S=P) Contraction coefficients', 'Coordinates of each shell', self.path)
        # else:
        #   self.contract_coeff = read_list_from_fchk('Contraction coefficients', 'Coordinates of each shell', self.path)

        self.coords_shell = read_list_from_fchk('Coordinates of each shell', 'Num ILSW', self.path)

    def get_ovlp(self):
        return build_ovlp(self)


def SS(mf, qns1, qns2, coords1, coords2, exp1, exp2):
    '''
    mol = objecte mol llegit
    mf = objecte mf llegit
    aa, bb = nombre quantic n,l,m. agafat de la funcio get_nlm(mf)
    AA, BB = Coordenades de cada shell. N'hi ha 3 per cada shell, van pel loop i de sota
    Prims1, Prims2 = Les primitives i, j respectivament, del total
    Entenc que pel cas del fitxer bz.fchk, que 7 primitives construeixen una shell.
    El que no entenc es com ho faig per ajuntar les 7 primitives juntes.
    Amb l'index i separo contribucions x, y i z i despres ho ajunto tot.
    help:)
    '''
    totalS = []
    EA = 0
    # print(len(qns1), 'qns1 len')
    # print(len(qns2), 'qns2 len')
    # print(len(coords1), 'coords1 len')
    # print(len(coords2), 'coords2 len')
    # print(exp1, 'exp1')
    # print(exp2, 'exp2')
    # Looping to each of the x, y and z axis
    # i nomes te en compte la coordenada cartesiana
    for ax in range(3):
        coord1 = coords1[ax]  # Component x, y or z for atom 1
        coord2 = coords2[ax]  # Component x, y or z for atom 2
        qn1 = qns1[ax]  # Quantum number for x, y or z axis
        qn2 = qns2[ax]  # Quantum number for x, y or z axis

        # for j in range(len(np.array(AA))):
        for iax in range(qn1 + 1):
            # for k in range(len(np.array(BB))):
            for jax in range(qn2 + 1):
                numprim = mf.n_prim_shells
                print('iax', iax, 'and jax', jax)

                Pax = (exp1 * coord1 + exp2 * coord2) / (exp1 + exp2)
                ct = np.sqrt((np.pi) / (exp1 + exp2))
                Piax = (Pax - coord1) ** (qn1 - iax)
                Pjax = (Pax - coord2) ** (qn2 - jax)
                comb_aa = comb(qn1, iax)
                comb_bb = comb(qn2, jax)
                f2 = factorial2(iax + jax - 1) / ((2 * (exp1 + exp2)) ** ((iax + jax) / 2))
        S_i = ct * comb_aa * comb_bb * f2 * Piax * Pjax
        totalS.append(S_i)

        print('shape total S', np.array(totalS).shape)
        Sij = np.prod(totalS)
        # Sij = np.prod(totalS)
        print('the value of the overlap in that place is', Sij)
        return Sij


def build_ovlp(mf):
    numprim = mf.n_prim_shells
    # print('numprim', numprim)
    nbasis = mf.nbas
    # print('nbasis', nbasis)
    mmax = len(mf.n_prim_per_shell)
    # print('mmax', mmax)
    natoms = mf.natm
    # print('natoms', natoms)
    nlm = get_nlm(mf)
    # print('nlm', np.shape(nlm))
    expp = mf.primitive_exps
    # print('expp', np.shape(expp))
    coefpb = mf.n_cont_shells
    # print('coefpb', coefpb)
    iptoat = mf.shell_to_atom_map
    # print('iptoat', np.shape(iptoat))

    # We reshape the data so we have it in form [x, y, z]

    # We get the quantum numbers n, l and m, to know the type of shell in each case
    coords = np.array(mf.coords_shell).reshape(-1, 3)
    exps = np.array(mf.primitive_exps)
    nshells = int(int(mf.n_prim_shells) / 3)
    # nshells = int(mol.nbas)
    S = np.zeros((nshells, nshells))
    for i in range(nshells):
        for j in range(nshells):
            p = 0
            scr = 0
            for p in range(int(mf.n_prim_per_shell[i])):
                print(p)
                scr += SS(mf, nlm[p], nlm[p], coords[i], coords[j], exps[i], exps[j])
                S[i, j] = scr

    print(S)
    return S

f = 'bz.fchk'

def comb(a, b):
    from math import factorial
    return factorial(a) / (factorial(b) * factorial(a - b))


def get_nlm(mf):
    shell_types = mf.shell_types
    numprims = len(shell_types)
    nprims_pershell = mf.n_prim_per_shell
    nlm = []
    count = 0
    for sh in range(numprims):
        s = shell_types[sh]
        pps = nprims_pershell[sh]
        if s == 0.0:
            for i in range(int(pps)):
                [nlm.append([0, 0, 0]) for i in range(int(s))]
                count += 1
        if s == 1.0 or s == -1.0:
            for i in range(int(pps)):
                for i in range(int(s)):
                    count += 3
                    nlm.append([1, 0, 0])
                    nlm.append([0, 1, 0])
                    nlm.append([0, 0, 1])
        if s == 2.0 or s == -2.0:
            for i in range(int(pps)):
                for i in range(int(s)):
                    count += 5
                    nlm.append([2, 0, 0])
                    nlm.append([0, 2, 0])
                    nlm.append([0, 0, 2])
                    nlm.append([1, 1, 0])
                    nlm.append([1, 0, 1])
                    nlm.append([0, 1, 1])
        # print('count for nlm', count)
    # print('pollas len nlm', len(nlm))
    return nlm
