"""
A slim FCHK loader that re-uses the existing FCHK parser and lets PySCF build
most internals. This file provides two lightweight classes:

- Mole2(path): parses FCHK via the existing reader, then creates a pyscf.gto.Mole
  from the geometry and the basis string found in the FCHK header. The pyscf
  Mole is built and exposed through `pyscf_mol` attribute; convenience attributes
  (_bas, _env, _atm, natm, nao, ...) are set from the pyscf object.

- MeanField2(path, mole2): builds a pyscf SCF object appropriate for the
  calculation (RHF/RKS/UHF/UKS) but DOES NOT run SCF. The MO coefficients are
  read from the FCHK and then permuted using `permute_aos_rows()` from
  esipy.tools to align Gaussian FCHK ordering with PySCF ordering.

This keeps parsing minimal and relies on PySCF to construct the low-level
_integral machinery (bas/env/atm) while preserving the FCHK reader logic.

This module intentionally makes minimal changes and is a compatibility shim.
"""

import os
import numpy as np
from pyscf import gto, scf

# Re-use parsing helpers from the original readfchk module
from esipy.tools import permute_aos_rows

# --- Embedded lightweight FCHK reader helpers (taken/adapted from readfchk.py) ---
_readers = {}

class FchkReader:
    def __init__(self, path):
        self.path = path
        with open(path, 'r') as f:
            self.lines = [ln.rstrip('\n') for ln in f]
        self.text = '\n'.join(self.lines)
        self._single_cache = {}
        self._list_cache = {}

    def find_first(self, prefix):
        if prefix in self._single_cache:
            return self._single_cache[prefix]
        for line in self.lines:
            if line.startswith(prefix):
                tokens = line.split()
                self._single_cache[prefix] = tokens
                return tokens
        return None

    def read_list(self, start, count=None):
        # If count not provided, parse integer from header line last token
        key = (start, count)
        if key in self._list_cache:
            return self._list_cache[key]
        out = []
        found = False
        start_idx = None
        header = ""
        for i, line in enumerate(self.lines):
            if line.startswith(start):
                start_idx = i
                header = line
                found = True
                break
        if not found:
            self._list_cache[key] = out
            return out
        # Determine count if not provided
        if count is None:
            last_tok = header.split()[-1]
            import re
            m = re.search(r"(\d+)", last_tok)
            if m:
                count = int(m.group(1))
        # Collect numbers from subsequent lines
        j = start_idx + 1
        while j < len(self.lines) and (count is None or len(out) < count):
            line = self.lines[j].strip()
            if not line:
                j += 1
                continue
            for tok in line.split():
                try:
                    out.append(float(tok))
                except Exception:
                    # ignore non-numeric tokens
                    pass
                if count is not None and len(out) >= count:
                    break
            j += 1
        self._list_cache[key] = out[:count] if count is not None else out
        return self._list_cache[key]


def _get_reader(path):
    path = os.path.abspath(path)
    if path in _readers:
        return _readers[path]
    r = FchkReader(path)
    _readers[path] = r
    return r


def read_from_fchk(to_read, path):
    r = _get_reader(path)
    res = r.find_first(to_read)
    return res if res is not None else []


def read_list_from_fchk(start, path):
    r = _get_reader(path)
    return r.read_list(start)


def read_level_theory(path):
    r = _get_reader(path)
    if len(r.lines) >= 2:
        return r.lines[1].split()
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

# --- End embedded reader helpers ---

class Mole2:
    """Light wrapper: parse FCHK using existing parser, build a PySCF Mole.

    Attributes:
      path: fchk path
      fchk: parsed `rchk.Mole2` instance (the original parser's object)
      pyscf_mol: the pyscf.gto.Mole built from the fchk info
      _bas/_env/_atm: taken from pyscf_mol after build()
      natm, nao, atomic_symbols, atomic_numbers, coord, basis, charge, spin
    """

    def __init__(self, path):
        self.path = path
        # Parse FCHK using local reader and build a small FCHK-side molecule
        self.fchk = FchkMolecule(path)

        # Extract basic metadata from the parsed FCHK object
        self.atomic_numbers = getattr(self.fchk, 'atomic_numbers', None)
        self.atomic_symbols = getattr(self.fchk, 'atomic_symbols', None)
        self.natm = int(getattr(self.fchk, 'natoms', getattr(self.fchk, 'natm', len(self.atomic_numbers))))
        self.charge = int(getattr(self.fchk, 'charge', 0))
        self.nalpha = int(getattr(self.fchk, 'nalpha', getattr(self.fchk, 'nalpha', 0)))
        self.nbeta = int(getattr(self.fchk, 'nbeta', getattr(self.fchk, 'nbeta', 0)))
        self.nelec = int(getattr(self.fchk, 'nelec', self.nalpha + self.nbeta))
        self.spin = int(getattr(self.fchk, 'spin', self.nalpha - self.nbeta))
        self.cart = bool(getattr(self.fchk, 'cart', False))
        coords = np.asarray(self.fchk.coord)
        self.coord = coords
        self.nummo = int(getattr(self.fchk, 'nummo', 0))
        self.numao = int(getattr(self.fchk, 'numao', 0))

        # basis name string from FCHK header (if present) -> use to instruct PySCF
        basis_tokens = read_level_theory(self.path)
        basis_name = basis_tokens[-1] if basis_tokens else None
        self.basis = basis_name
        self.fchk_basis_arrays = {
            'mssh': getattr(self.fchk, 'mssh', None),
            'mnsh': getattr(self.fchk, 'mnsh', None),
            'iatsh': getattr(self.fchk, 'iatsh', None),
            'expsh': getattr(self.fchk, 'expsh', None),
            'c1': getattr(self.fchk, 'c1', None),
            'c2': getattr(self.fchk, 'c2', None),
            'ncshell': getattr(self.fchk, 'ncshell', None),
        }
        self.basis = make_basis(self)

        # Building the PySCF Mole from the FCHK info
        pyscf_mol = gto.M()
        atom_list = []
        for sym, xyz in zip(self.atomic_symbols, coords):
            atom_list.append((sym, (float(xyz[0]), float(xyz[1]), float(xyz[2]))))

        pyscf_mol.atom = atom_list
        pyscf_mol.basis = self.basis

        # preserve spin/charge
        pyscf_mol.charge = int(self.charge)
        pyscf_mol.spin = int(self.spin) if hasattr(self, 'spin') else 0
        pyscf_mol.cart = getattr(self.fchk, 'cart', False)
        pyscf_mol.unit = 'Bohr'
        pyscf_mol.verbose = 0
        # avoid symmetry auto-detection that may reorder things
        pyscf_mol.symmetry = False

        pyscf_mol.build()

        self.pyscf_mol = pyscf_mol

        # PySCF-internal bas/env/atm (may be in PySCF ordering)
        self._bas = getattr(pyscf_mol, '_bas', None)
        self._env = getattr(pyscf_mol, '_env', None)
        self._atm = getattr(pyscf_mol, '_atm', None)
        self.nao = int(getattr(pyscf_mol, 'nao_nr', lambda: pyscf_mol.nao)()) if hasattr(pyscf_mol, 'nao_nr') else int(getattr(pyscf_mol, 'nao', 0))
        self.nbas = int(getattr(pyscf_mol, 'nbas', 0))
        # keep a reference to original fchk arrays in case they are needed

    # Convenience methods used by code expecting Mole-like API
    def atom_coords(self):
        return self.pyscf_mol.atom_coords()

    def atom_symbol(self, i):
        return self.pyscf_mol.atom_symbol(i)

    def aoslice_by_atom(self, ao_loc=None):
        return self.pyscf_mol.aoslice_by_atom()

    def ao_loc_nr(self):
        return self.pyscf_mol.ao_loc_nr()

    def intor(self, intor_name, comp=1):
        return self.pyscf_mol.intor(intor_name, comp=comp)

    def intor_symmetric(self, intor_name, comp=1):
        return self.pyscf_mol.intor_symmetric(intor_name, comp=comp)

    def bas_atom(self, ib):
        return self.pyscf_mol.bas_atom(ib)

    def bas_angular(self, ib):
        return self.pyscf_mol.bas_angular(ib)

    def bas_nctr(self, ib):
        return self.pyscf_mol.bas_nctr(ib)

    def atom_nelec_core(self, ia):
        return self.pyscf_mol.atom_nelec_core(ia)

    def atom_pure_symbol(self, ia):
        return self.pyscf_mol.atom_pure_symbol(ia)

    def nao_nr(self):
        return self.pyscf_mol.nao_nr()

class MeanField2:
    """Thin wrapper that exposes SCF object built by PySCF, but uses MO coeffs
    read from the FCHK file (permuted to PySCF AO order).
    """

    def __init__(self, path, mole2: Mole2):
        self.path = path
        self.mole2 = mole2
        self.mol = mole2.pyscf_mol

        # Determine whether closed or open shell from FCHK
        self.nbasis = self.mole2.nao
        self.natm = self.mole2.natm
        self.cart = getattr(self.mole2.fchk, 'cart', False)
        self.nao = self.mole2.numao
        self.nalpha = int(getattr(self.mole2.fchk, 'nalpha', 0))
        self.nbeta = int(getattr(self.mole2.fchk, 'nbeta', 0))
        self.nummo = int(getattr(self.mole2.fchk, 'nummo', 0))
        self.numao = int(getattr(self.mole2.fchk, 'numao', 0))
        nalpha = self.nalpha
        nbeta = self.nbeta
        nocc_alpha = nalpha // 2.
        nocc_beta = nbeta // 2.
        self.e_tot = float(getattr(self.mole2.fchk, 'total_energy', 0.0))
        if nalpha == nbeta:
            # Restricted
            self._scf = scf.RHF(self.mol)
            self.__name__ = "RHF"
        else:
            self._scf = scf.UHF(self.mol)
            self.__name__ = "UHF"


        # Read MO coefficients from FCHK (Gaussian order) and reshape
        wf = 'rest' if nalpha == nbeta else 'unrest'
        if wf == 'rest':
            mo_flat = read_list_from_fchk('Alpha MO coefficients', path)
            if len(mo_flat) == 0:
                raise RuntimeError('No MO coefficients found in FCHK')
            mo_arr = np.array(mo_flat, dtype=float).reshape(self.nummo, self.numao).T
            mo_arr = permute_aos_rows(mo_arr, self.mole2)
            self.mo_coeff = mo_arr
            self.mo_occ = np.array([2.] * (int(self.nalpha + self.nbeta) // 2) + [0.] * (self.nao - (int(self.nalpha + self.nbeta) // 2)))
        else:
            # For simplicity, only handle alpha coefficients then beta similarly
            mo_flat_a = read_list_from_fchk('Alpha MO coefficients', path)
            mo_flat_b = read_list_from_fchk('Beta MO coefficients', path)
            nummo = int(getattr(self.mole2.fchk, 'nummo', len(mo_flat_a) // int(self.mole2.nao)))
            numao = int(self.mole2.nao)
            mo_arr_a = np.array(mo_flat_a, dtype=float).reshape(nummo, numao).T
            mo_arr_b = np.array(mo_flat_b, dtype=float).reshape(nummo, numao).T
            mo_arr_a = permute_aos_rows(mo_arr_a, self.mole2)
            mo_arr_b = permute_aos_rows(mo_arr_b, self.mole2)
            mo_arr = [mo_arr_a, mo_arr_b]
            self.mo_coeff = mo_arr
            self.mo_occ = [1] * nocc_alpha + [1] * nocc_beta + [0] * (
                    len(self.alpha_orbital_energies) + self.nao - nocc_alpha - nocc_beta)

    def make_rdm1(self, ao_repr=True):
        # Simple density from mo_coeff and mo_occ if available (RHF case)
        if self.mo_coeff is None:
            return None
        if isinstance(self.mo_coeff, list):
            # UHF
            ca, cb = self.mo_coeff
            occ_a = np.asarray(self.mo_occ[0]) if isinstance(self.mo_occ, (list, tuple, np.ndarray)) else np.ones(ca.shape[1])
            occ_b = np.asarray(self.mo_occ[1]) if isinstance(self.mo_occ, (list, tuple, np.ndarray)) else np.ones(cb.shape[1])
            Da = ca[:, :np.count_nonzero(occ_a)].dot(ca[:, :np.count_nonzero(occ_a)].T)
            Db = cb[:, :np.count_nonzero(occ_b)].dot(cb[:, :np.count_nonzero(occ_b)].T)
            return np.array([Da, Db])
        else:
            occ = np.asarray(self.mo_occ) if self.mo_occ is not None else None
            if occ is None:
                # try to infer closed-shell occupancy: highest nelec/2 orbitals occupied
                nocc = int(getattr(self.mole2, 'nelec', 0) // 2)
                C = self.mo_coeff[:, :nocc]
                return 2.0 * C.dot(C.T)
            else:
                occupied = np.where(np.asarray(occ) > 0)[0]
                C = self.mo_coeff[:, occupied]
                if C.size == 0:
                    return np.zeros((self.nao, self.nao))
                return 2.0 * C.dot(C.T)

    def get_ovlp(self):
        return self.mol.intor_symmetric('int1e_ovlp')

# Custom object that just reads the FCHK file
class FchkMolecule:
    def __init__(self, path):
        self.path = path
        # basic scalars
        self.nalpha = int(read_from_fchk('Number of alpha electrons', path)[-1])
        self.nbeta = int(read_from_fchk('Number of beta electrons', path)[-1])
        self.natoms = int(read_from_fchk('Number of atoms', path)[-1])
        self.ncshell = int(read_from_fchk('Number of contracted shells', path)[-1])

        self.atomic_numbers = [int(i) for i in read_list_from_fchk('Atomic numbers', path)]
        self.atomic_symbols = read_atomic_symbols(self.atomic_numbers)
        self.coord = np.array(read_list_from_fchk('Current cartesian coordinates', path)).reshape(self.natoms, 3)

        # shell arrays
        self.mssh = [int(i) for i in read_list_from_fchk('Shell types', path)]
        self.mnsh = [int(i) for i in read_list_from_fchk('Number of primitives per shell', path)]
        self.iatsh = [int(i) for i in read_list_from_fchk('Shell to atom map', path)]
        self.expsh = read_list_from_fchk('Primitive exponents', path)

        # contraction coeffs
        with open(path, 'r') as f:
            filetext = f.read()
        if 'P(S=P)' in filetext:
            self.c1 = read_list_from_fchk('Contraction coefficients', path)
            self.c2 = read_list_from_fchk('P(S=P) Contraction coefficients', path)
        else:
            self.c1 = read_list_from_fchk('Contraction coefficients', path)
            self.c2 = None

        # numbers
        self.nummo = int(read_from_fchk('Number of independent functions', path)[-1])
        self.numao = int(read_from_fchk('Number of basis functions', path)[-1])

def make_basis(mf):
    """
    Generate a basis dict from a PySCF-derived mf object.
    Output format: {'C': [[l, [exp, c1, c2...], ...], ...], ...}
    """
    atomic_symbols = mf.atomic_symbols
    ncshell = mf.fchk.ncshell
    mssh = mf.fchk.mssh
    mnsh = mf.fchk.mnsh
    iatsh = mf.fchk.iatsh
    expsh = mf.fchk.expsh
    c1 = mf.fchk.c1
    c2 = mf.fchk.c2 if hasattr(mf.fchk, 'c2') else None
    done_shells = {}

    exp_idx = 0
    coeff_idx = 0

    first_atom = {}
    for atom_idx, sym in enumerate(mf.atomic_symbols):
        if sym not in first_atom:
            first_atom[sym] = atom_idx

    # --- Main Parsing Loop (unchanged from your version) ---
    for i in range(ncshell):
        l_raw = mssh[i]
        atom_idx = iatsh[i] - 1
        sym = atomic_symbols[atom_idx]
        n_prim = mnsh[i]

        # Skip appearance if not the first atom of this element type
        if atom_idx != first_atom[sym]:
            # Advance exp/coeff pointers by same logic as your original code
            exp_idx += n_prim
            if l_raw == -1:
                coeff_idx += n_prim
            else:
                # deduce number of contractions for non-SP shells
                if i == ncshell - 1:
                    n_contr_to_skip = (len(c1) - coeff_idx) // n_prim
                else:
                    remaining_prims = sum(mnsh[k] for k in range(i + 1, ncshell))
                    remaining_coeffs = len(c1) - coeff_idx - remaining_prims
                    n_contr_to_skip = remaining_coeffs // n_prim if n_prim else 0
                coeff_idx += n_prim * n_contr_to_skip
            continue

        if sym not in done_shells:
            done_shells[sym] = []

        primitives = []
        if l_raw == -1:
            # SP shell: c1 -> S, c2 -> P; we store tuples (exponent, c_s, c_p)
            for _ in range(n_prim):
                exponent = expsh[exp_idx]
                coef_s = c1[coeff_idx]
                coef_p = c2[coeff_idx] if c2 is not None else 0.0
                primitives.append((exponent, coef_s, coef_p))
                exp_idx += 1
                coeff_idx += 1
            done_shells[sym].append({"l": -1, "primitives": primitives})
        else:
            # Regular shell: deduce n_contr as you did
            if i == ncshell - 1:
                n_contr = (len(c1) - coeff_idx) // n_prim if n_prim else 0
            else:
                remaining_prims = sum(mnsh[k] for k in range(i + 1, ncshell))
                remaining_coeffs = len(c1) - coeff_idx - remaining_prims
                n_contr = (remaining_coeffs // n_prim) if n_prim else 0

            coeffs_flat = c1[coeff_idx : coeff_idx + n_prim * n_contr]
            for prim_idx in range(n_prim):
                exponent = expsh[exp_idx + prim_idx]
                coefs = [coeffs_flat[j * n_prim + prim_idx] for j in range(n_contr)]
                primitives.append((exponent, coefs))
            exp_idx += n_prim
            coeff_idx += n_prim * n_contr
            done_shells[sym].append({"l": abs(l_raw), "primitives": primitives})

    # --- TRANSFORMATION: build per-l lists preserving original encounter order ---
    final_basis_dict = {}
    for sym, shells in done_shells.items():
        # lists_by_l will gather shells in encounter order per angular momentum
        lists_by_l = {}  # maps l -> list of shells (each shell is [l, [exp, coefs...], ...])
        max_l = 0

        for shell_data in shells:
            l = shell_data["l"]
            primitives_data = shell_data["primitives"]

            if l == -1:
                # (In your parse l==-1 shells were stored as tuple (exp, c_s, c_p))
                # create S-shell (l=0) and P-shell (l=1) and append them to their lists
                s_shell = [0]
                p_shell = [1]
                for exp, cs, cp in primitives_data:
                    s_shell.append([exp, cs])
                    p_shell.append([exp, cp])
                lists_by_l.setdefault(0, []).append(s_shell)
                lists_by_l.setdefault(1, []).append(p_shell)
                max_l = max(max_l, 1)
            else:
                # regular shell: primitives_data entries are (exp, [coefs...])
                new_shell = [l]
                for exp, coefs in primitives_data:
                    new_shell.append([exp] + (coefs if isinstance(coefs, list) else [coefs]))
                lists_by_l.setdefault(l, []).append(new_shell)
                max_l = max(max_l, l)

        # concatenate in l order: 0,1,2,...
        final_list = []
        for L in range(max_l + 1):
            if L in lists_by_l:
                final_list.extend(lists_by_l[L])

        final_basis_dict[sym] = final_list

    return final_basis_dict

def readfchk(path):
    """Convenience function to read FCHK and return Mole2 and MeanField2 objects.
    """
    mol2 = Mole2(path)
    mf2 = MeanField2(path, mol2)
    return mol2, mf2
