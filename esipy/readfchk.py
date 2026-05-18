import sys
from pyscf import scf
import os
import numpy as np
import scipy.linalg
from pyscf import gto, scf
from pyscf.data import elements
from pyscf.lo import orth

from esipy.tools import permute_aos_rows

def read_from_fchk(to_read, path):
    """
    Reads a line from a Gaussian FCHK file.
    """
    with open(path, 'r') as f:
        for line in f:
            if to_read in line:
                return line.split()
    return []


def read_list_from_fchk(start, path):
    """
    Reads a list from a Gaussian FCHK file.
    """
    with open(path, 'r') as f:
        found = False
        out = []
        count = None
        for line in f:
            if start in line:
                found = True
                try:
                    count = int(line.split()[-1])
                except:
                    count = None
                continue
            if found:
                if any(c.isalpha() for c in line) and not 'E' in line.upper():
                    break
                for tok in line.split():
                    try:
                        out.append(float(tok))
                    except:
                        pass
                if count is not None and len(out) >= count:
                    break
        return out


def read_level_theory(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        if len(lines) >= 2:
            return lines[1].split()
    return []


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


class Mole2:
    """Builds a PySCF Mole object from a Gaussian FCHK file.
    """

    def __init__(self, path):
        self.path = path
        self.fchk = FchkMolecule(path)

        self.pyscf_mol = gto.Mole()
        self.pyscf_mol.atom = [[self.fchk.atomic_symbols[i], self.fchk.coord[i]] for i in range(self.fchk.natoms)]
        self.pyscf_mol.unit = 'Bohr'
        self.pyscf_mol.charge = int(self.fchk.charge)
        self.pyscf_mol.spin = int(self.fchk.spin)
        
        # Determine basis set
        basis_tokens = read_level_theory(self.path)
        basis_name = basis_tokens[-1] if basis_tokens else None
        self.pyscf_mol.basis = basis_name
        self.pyscf_mol.verbose = 0

        # Mapping for AO permutation
        self.fchk_basis_arrays = self.fchk.fchk_basis_arrays
        self.numao = self.fchk.numao
        self.nummo = self.fchk.nummo
        self.natm = self.fchk.natoms

    def build(self, *args, **kwargs):
        kwargs.setdefault('verbose', 0)
        self.pyscf_mol.build(*args, **kwargs)
        return self

    def __getattr__(self, name):
        return getattr(self.pyscf_mol, name)


class MeanField2:
    """
    Builds a PySCF MeanField object from a Gaussian FCHK file.
    """

    def __init__(self, path, mole2: Mole2):
        self.path = path
        self.mole2 = mole2
        self.mol = mole2.pyscf_mol
        self.pyscf_mol = mole2.pyscf_mol

        self.nao = self.mole2.numao
        self.nummo = self.mole2.nummo

        self.nalpha = self.mole2.fchk.nalpha
        self.nbeta = self.mole2.fchk.nbeta
        self.e_tot = self.mole2.fchk.e_tot
        self.charge = self.mole2.fchk.charge

        if self.mole2.fchk.unrestricted:
            self._scf = scf.UHF(self.mol)
            self.__name__ = "UHF"
        else:
            self._scf = scf.RHF(self.mol)
            self.__name__ = "RHF"

        # Standard MO reading
        mo_flat_a = read_list_from_fchk('Alpha MO coefficients', path)
        if len(mo_flat_a) == 0:
            raise RuntimeError('No MO coefficients found in FCHK')
        
        mo_arr_a = np.array(mo_flat_a, dtype=float).reshape(self.nummo, self.nao).T
        self.mo_coeff_a = permute_aos_rows(mo_arr_a, self.mole2)
        
        mo_flat_b = read_list_from_fchk('Beta MO coefficients', path)
        if len(mo_flat_b) > 0:
            mo_arr_b = np.array(mo_flat_b, dtype=float).reshape(self.nummo, self.nao).T
            self.mo_coeff_b = permute_aos_rows(mo_arr_b, self.mole2)
            self.mo_coeff = [self.mo_coeff_a, self.mo_coeff_b]
            self.mo_occ = [np.zeros(self.nummo), np.zeros(self.nummo)]
            self.mo_occ[0][:self.nalpha] = 1.0
            self.mo_occ[1][:self.nbeta] = 1.0
            if not isinstance(self._scf, scf.uhf.UHF):
                self._scf = scf.UHF(self.mol)
                self.__name__ = "UHF"
        else:
            self.mo_coeff = self.mo_coeff_a
            nocc = (self.nalpha + self.nbeta) // 2
            self.mo_occ = np.zeros(self.nummo)
            self.mo_occ[:nocc] = 2.0

        self._scf.mo_coeff = self.mo_coeff
        self._scf.mo_occ = self.mo_occ
        self._scf.e_tot = self.e_tot

    def __getattr__(self, name):
        return getattr(self._scf, name)


class FchkMolecule:
    def __init__(self, path):
        self.unrestricted = False
        res_spin = read_from_fchk('Multiplicity', path)
        if res_spin:
            self.spin = int(res_spin[-1]) - 1
        else:
            self.spin = 0

        res_nalpha = read_from_fchk('Number of alpha electrons', path)
        self.nalpha = int(res_nalpha[-1]) if res_nalpha else 0
        res_nbeta = read_from_fchk('Number of beta electrons', path)
        self.nbeta = int(res_nbeta[-1]) if res_nbeta else 0
        
        if self.spin > 0 or self.nalpha != self.nbeta:
            self.unrestricted = True

        self.natoms = int(read_from_fchk('Number of atoms', path)[-1])
        
        # Bug fix: Charge can be a list or a single number in FCHK
        res_charge = read_from_fchk('Charge', path)
        if isinstance(res_charge, list) and len(res_charge) > 0:
            try:
                self.charge = int(res_charge[-1])
            except ValueError:
                self.charge = 0
        elif res_charge:
            self.charge = int(res_charge)
        else:
            self.charge = 0

        self.atomic_numbers = [int(i) for i in read_list_from_fchk('Atomic numbers', path)]
        self.atomic_symbols = read_atomic_symbols(self.atomic_numbers)
        self.coord = np.array(read_list_from_fchk('Current cartesian coordinates', path)).reshape(self.natoms, 3)
        
        res_etot = read_from_fchk('Total Energy', path)
        self.e_tot = float(res_etot[-1]) if res_etot else 0.0

        self.nummo = int(read_from_fchk('Number of independent functions', path)[-1])
        self.numao = int(read_from_fchk('Number of basis functions', path)[-1])

        self.fchk_basis_arrays = {
            'mssh': [int(i) for i in read_list_from_fchk('Shell types', path)],
            'mnsh': [int(i) for i in read_list_from_fchk('Number of primitives per shell', path)],
            'iatsh': [int(i) for i in read_list_from_fchk('Shell to atom map', path)],
        }


def readfchk(path):
    """Convenience function to read FCHK and return Mole2 and MeanField2 objects.
    """
    print(" | Reading FCHK file:", path)
    mol2 = Mole2(path)
    mol2.build()
    mf2 = MeanField2(path, mol2)
    return mol2, mf2._scf
