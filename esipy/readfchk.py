import re
import numpy as np

from pyscf import scf
from pyscf.gto import basis

# Simple in-memory reader cache to avoid reopening and reparsing the fchk file multiple times
_readers = {}

class FchkReader:
    """Lightweight parser for formatted checkpoint (.fchk) files with caching of parsed (start_label,end_label) lists and single-line lookups.

    Stores lines and a small cache of parsed (start_label,end_label) lists and single-line lookups.
    """
    def __init__(self, path):
        self.path = path
        # Read file once into memory
        with open(path, 'r') as f:
            self.lines = [ln.rstrip('\n') for ln in f]
        self.text = '\n'.join(self.lines)
        self._single_cache = {}   # prefix -> split tokens
        self._list_cache = {}     # (start,end) -> list of floats

    def find_first(self, prefix):
        if prefix in self._single_cache:
            return self._single_cache[prefix]
        for line in self.lines:
            if line.startswith(prefix):
                tokens = line.split()
                self._single_cache[prefix] = tokens
                return tokens
        return None

    def read_list(self, start, end):
        key = (start, end)
        if key in self._list_cache:
            return self._list_cache[key]
        out = []
        found = False
        for line in self.lines:
            if not found:
                if line.startswith(start):
                    found = True
                    # Skip the starting line entirely (the numeric payload follows on subsequent lines)
                    continue
            else:
                if end in line:
                    break
                # collect numeric tokens
                for tok in line.split():
                    try:
                        out.append(float(tok))
                    except Exception:
                        pass
        self._list_cache[key] = out
        return out

    def level_theory(self):
        # emulates previous read_level_theory which returned second line tokens after the first
        # We'll return tokens from the second line of the file if available
        if len(self.lines) >= 2:
            return self.lines[1].split()
        return []

    def has_symmetry(self):
        # previous implementation skipped 9 lines then checked the next
        if len(self.lines) > 10:
            line = self.lines[10].split()
            for w in line:
                if 'symm' in w.lower():
                    return True
        # fallback: search for 'symm' anywhere
        return 'symm' in self.text.lower()

    def read_contract_coeff(self):
        # read contraction coefficients from the single starting line like original
        # the original function was somewhat broken; we implement a simple version
        out = []
        for i, line in enumerate(self.lines):
            if line.startswith('Contraction coefficients'):
                parts = line.split()
                # last token was the count in original code
                try:
                    s = int(parts[-1])
                except Exception:
                    s = None
                # collect numeric tokens in the line
                for tok in parts:
                    try:
                        val = float(tok)
                        out.append(val)
                    except Exception:
                        pass
                # if we didn't get enough and next lines contain values, continue
                j = i + 1
                while s is None or len(out) < s:
                    if j >= len(self.lines):
                        break
                    for tok in self.lines[j].split():
                        try:
                            out.append(float(tok))
                        except Exception:
                            pass
                    j += 1
                if s is not None:
                    out = out[:s]
                return out
        return out

class Overlapper:
    """Tiny helper that centralizes overlap caching.

    Usage: Overlapper(obj).get()
    - If obj._ovlp (or obj.mol._ovlp) exists, return it.
    - Otherwise call process_basis(obj) and build_ovlp(obj), store result
      on obj._ovlp and (if present) obj.mol._ovlp, and return it.
    """
    def __init__(self, obj):
        self.obj = obj
        self._ovlp = None

    def get(self):
        # prefer mf-level cache
        if getattr(self.obj, '_ovlp', None) is not None:
            return self.obj._ovlp
        # prefer molecule-level cache if available
        if hasattr(self.obj, 'mol') and getattr(self.obj.mol, '_ovlp', None) is not None:
            return self.obj.mol._ovlp

        # Not cached: ensure basis-derived attributes exist and compute
        if not getattr(self.obj, '_processed', False):
            process_basis(self.obj)
            # mark both the object and its mol as processed so subsequent calls skip reprocessing
            try:
                self.obj._processed = True
            except Exception:
                pass
            if hasattr(self.obj, 'mol'):
                try:
                    self.obj.mol._processed = True
                except Exception:
                    pass
        mat = build_ovlp(self.obj)

        try:
            self.obj._ovlp = mat
        except Exception:
            pass
        # also store on molecule (so Mole2 and MeanField2 share the cache)
        if hasattr(self.obj, 'mol'):
            try:
                self.obj.mol._ovlp = mat
            except Exception:
                pass
        return mat



def _get_reader(path):
    path = os.path.abspath(path)
    if path in _readers:
        return _readers[path]
    r = FchkReader(path)
    _readers[path] = r
    return r


def readfchk(filename):
    """
    Reads the information from a fchk file.
    Args:
        filename: Name of the fchk file.

    Returns:
        Mole object and MeanField object.
    """
    mol = Mole2(filename)
    mf = MeanField2(filename, mol)
    return mol, mf


def read_from_fchk(to_read, path):
    r = _get_reader(path)
    res = r.find_first(to_read)
    return res


def read_level_theory(path):
    r = _get_reader(path)
    return r.level_theory()


def read_symmetry(path):
    r = _get_reader(path)
    return r.has_symmetry()


def read_contract_coeff(path):
    r = _get_reader(path)
    return r.read_contract_coeff()


def read_list_from_fchk(start, path):
    """
    Read a numeric list from an FCHK file given the starting label.
    Behavior: finds the line that starts with `start`, expects the last token on that
    same line to contain the integer count (e.g. "N= 69696" or just "69696"), and
    then reads subsequent lines collecting floats until the requested count is reached.
    Returns a Python list of floats (length == count when successful).
    """
    r = _get_reader(path)
    # find index of start line
    start_idx = None
    for i, line in enumerate(r.lines):
        if line.startswith(start):
            start_idx = i
            header = line
            break
    if start_idx is None:
        return []
    # parse integer count from last token of header line
    last_tok = header.split()[-1]
    m = re.search(r"(\d+)", last_tok)
    if not m:
        # fallback: if no explicit count found, try to parse numeric tokens on same line
        nums = []
        for tok in header.split()[1:]:
            try:
                nums.append(float(tok))
            except Exception:
                pass
        return nums
    count = int(m.group(1))

    nums = []
    j = start_idx + 1
    while j < len(r.lines) and len(nums) < count:
        line = r.lines[j].strip()
        if not line:
            j += 1
            continue
        for tok in line.split():
            try:
                nums.append(float(tok))
            except Exception:
                # allow tokens like 'D' or other markers to be ignored
                pass
            if len(nums) >= count:
                break
        j += 1
    return nums[:count]


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

def make_bas_env(self):
    import numpy as np
    from pyscf.gto.mole import gto_norm

    # ---- env header + coords (unchanged) ----
    env = [0.0] * 20
    coords = getattr(self, 'coord', None)
    for (x, y, z) in coords:
        env.extend([x, y, z, 0.0])

    prim_ptr = [0]
    for n in self.mnsh:
        prim_ptr.append(prim_ptr[-1] + int(n))
    iatsh = [int(i) - 1 for i in self.iatsh]

    bas = []
    env_idx = len(env)

    # ---- new: intern table for unique blocks ----
    # maps (tuple(exps), tuple(coeffs)) → ptr_exp
    intern_exp = {}
    intern_coeff = {}

    def intern_array(arr, table):
        """Return pointer into env for this array,
        adding it only if not already present."""
        key = tuple(arr.tolist())
        if key in table:
            return table[key]
        nonlocal env_idx
        ptr = env_idx
        env.extend(arr.tolist())
        env_idx += len(arr)
        table[key] = ptr
        return ptr

    # ---- MAIN LOOP ----
    for ish in range(self.ncshell):
        lraw = int(self.mssh[ish])
        ia = iatsh[ish]
        p0, p1 = prim_ptr[ish], prim_ptr[ish + 1]
        nprim = p1 - p0
        exps = np.asarray(self.expsh[p0:p1], float)

        # ---------- SP shells ----------
        if lraw == -1:
            # S part
            arr_s = np.asarray(self.c1[p0:p1], float)
            if arr_s.size:
                nctr_s = arr_s.size // nprim
                cs = arr_s.reshape(nprim, nctr_s) * gto_norm(0, exps).reshape(-1, 1)
                ptr_exp = intern_array(exps, intern_exp)
                ptr_coeff = intern_array(cs.flatten(order="F"), intern_coeff)
                bas.append([ia, 0, nprim, nctr_s, 0, ptr_exp, ptr_coeff, 0])

            # P part
            arr_p = np.asarray(self.c2[p0:p1], float)
            if arr_p.size:
                nctr_p = arr_p.size // nprim
                cp = arr_p.reshape(nprim, nctr_p) * gto_norm(1, exps).reshape(-1, 1)
                ptr_exp = intern_array(exps, intern_exp)
                ptr_coeff = intern_array(cp.flatten(order="F"), intern_coeff)
                bas.append([ia, 1, nprim, nctr_p, 0, ptr_exp, ptr_coeff, 0])

            continue

        # ---------- normal shells ----------
        l = abs(lraw)
        arr = np.asarray(self.c1[p0:p1], float)
        nctr = arr.size // nprim if arr.size else 1
        cs = arr.reshape(nprim, nctr) * gto_norm(l, exps).reshape(-1, 1)

        ptr_exp = intern_array(exps, intern_exp)
        ptr_coeff = intern_array(cs.flatten(order="F"), intern_coeff)
        bas.append([ia, l, nprim, nctr, 0, ptr_exp, ptr_coeff, 0])

    _bas = np.array(bas, dtype=np.int32)
    _env = np.array(env, dtype=np.float64)
    return _bas, _env


def make_atm(mol):
    """
    Constructs _atm matching the _env layout created above.
    Assumes coords start at env[20].
    """
    _atm = []
    # PySCF standard offset for atoms in env
    PTR_ENV_START = 20

    for i, Z in enumerate(mol.atomic_numbers):
        # Calculate pointers into the env array
        # Each atom took 4 slots in env [x, y, z, 0.0]
        ptr_coord = PTR_ENV_START + (i * 4)

        # PySCF _atm row structure:
        # [Charge, ptr_coord, nucl_model, ptr_zeta, 0, 0]
        # ptr_zeta points to the 4th element of the coord block (the 0.0 or actual Z)
        # Standard PySCF often sets nucl_model=1 (point charge)

        atom_row = [
            int(Z),  # Nuclear Charge
            ptr_coord,  # Pointer to X coordinate
            1,  # Nuclear model (1 = point charge)
            ptr_coord + 3,  # Pointer to Zeta (often stored in the 4th float of coord block)
            0,
            0
        ]
        _atm.append(atom_row)

    return np.array(_atm, dtype=np.intc)

def make_basis(mf):
    """
    Generate a basis dict from a PySCF-derived mf object.
    Output format: {'C': [[l, [exp, c1, c2...], ...], ...], ...}
    """
    done_shells = {}

    exp_idx = 0
    coeff_idx = 0

    first_atom = {}
    for atom_idx, sym in enumerate(mf.atomic_symbols):
        if sym not in first_atom:
            first_atom[sym] = atom_idx

    # --- Main Parsing Loop (unchanged from your version) ---
    for i in range(mf.ncshell):
        l_raw = mf.mssh[i]
        atom_idx = mf.iatsh[i] - 1
        sym = mf.atomic_symbols[atom_idx]
        n_prim = mf.mnsh[i]

        # Skip appearance if not the first atom of this element type
        if atom_idx != first_atom[sym]:
            # Advance exp/coeff pointers by same logic as your original code
            exp_idx += n_prim
            if l_raw == -1:
                coeff_idx += n_prim
            else:
                # deduce number of contractions for non-SP shells
                if i == mf.ncshell - 1:
                    n_contr_to_skip = (len(mf.c1) - coeff_idx) // n_prim
                else:
                    remaining_prims = sum(mf.mnsh[k] for k in range(i + 1, mf.ncshell))
                    remaining_coeffs = len(mf.c1) - coeff_idx - remaining_prims
                    n_contr_to_skip = remaining_coeffs // n_prim if n_prim else 0
                coeff_idx += n_prim * n_contr_to_skip
            continue

        if sym not in done_shells:
            done_shells[sym] = []

        primitives = []
        if l_raw == -1:
            # SP shell: c1 -> S, c2 -> P; we store tuples (exponent, c_s, c_p)
            for _ in range(n_prim):
                exponent = mf.expsh[exp_idx]
                coef_s = mf.c1[coeff_idx]
                coef_p = mf.c2[coeff_idx] if mf.c2 is not None else 0.0
                primitives.append((exponent, coef_s, coef_p))
                exp_idx += 1
                coeff_idx += 1
            done_shells[sym].append({"l": -1, "primitives": primitives})
        else:
            # Regular shell: deduce n_contr as you did
            if i == mf.ncshell - 1:
                n_contr = (len(mf.c1) - coeff_idx) // n_prim if n_prim else 0
            else:
                remaining_prims = sum(mf.mnsh[k] for k in range(i + 1, mf.ncshell))
                remaining_coeffs = len(mf.c1) - coeff_idx - remaining_prims
                n_contr = (remaining_coeffs // n_prim) if n_prim else 0

            coeffs_flat = mf.c1[coeff_idx : coeff_idx + n_prim * n_contr]
            for prim_idx in range(n_prim):
                exponent = mf.expsh[exp_idx + prim_idx]
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

class Mole2():
    def __init__(self, path):
        self.path = path
        self.basis = read_level_theory(self.path)[-1]
        self._ovlp = None
        self._read_fchk = True
        self._reorder = True
        self.nalpha = int(read_from_fchk('Number of alpha electrons', self.path)[-1])
        self.nbeta = int(read_from_fchk('Number of beta electrons', self.path)[-1])
        self.numao = int(read_from_fchk('Number of basis functions', self.path)[-1])
        self.nummo = int(read_from_fchk('Number of independent functions', self.path)[-1])
        self.charge = int(read_from_fchk('Charge', self.path)[-1])
        self.spin = int(self.nalpha) - int(self.nbeta)
        self.symmetry = read_symmetry(self.path)
        self.nelec = int(read_from_fchk('Number of electrons', self.path)[-1])
        self.nelectron = self.nelec
        self.charge = int(read_from_fchk('Charge', self.path)[-1])
        self.natoms = int(read_from_fchk('Number of atoms', self.path)[-1])
        self.natm = self.natoms
        self.atomic_numbers = [int(i) for i in read_list_from_fchk('Atomic numbers', self.path)]
        self.atomic_symbols = read_atomic_symbols(self.atomic_numbers)
        self.atom = self.make_atom()
        self.nbasis = int(read_from_fchk('Number of basis functions', self.path)[-1])
        self.nao = self.nbasis
        self.dcart = int(read_from_fchk("Pure/Cartesian d shells", self.path)[-1])
        self.fcart = int(read_from_fchk("Pure/Cartesian f shells", self.path)[-1])
        self.cart = (self.dcart != 0)
        self.verbose = 0
        self.numprim = int(read_from_fchk('Number of primitive shells', self.path)[-1])
        self.mssh = read_list_from_fchk('Shell types', self.path)
        self.mssh = [int(i) for i in self.mssh]
        # Number of primitives per shell
        self.mnsh = read_list_from_fchk('Number of primitives per shell', self.path)
        self.mnsh = [int(i) for i in self.mnsh]
        # Shell to atom map
        self.iatsh = read_list_from_fchk('Shell to atom map', self.path)
        self.iatsh = [int(i) for i in self.iatsh]
        # Primitive exponents
        self.expsh = read_list_from_fchk('Primitive exponents', self.path)
        # Current cartesian coordinates
        coords = read_list_from_fchk('Current cartesian coordinates', self.path)
        self.coord = np.array(coords).reshape(int(self.natoms), 3)

        with open(path, 'r') as file:
            if 'P(S=P)' in file.read():
                self.c1 = read_list_from_fchk('Contraction coefficients', self.path)
                self.c2 = read_list_from_fchk('P(S=P) Contraction coefficients', self.path)

            else:
                self.c1 = read_list_from_fchk('Contraction coefficients', self.path)
                self.c2 = None

        self.ncshell = int(read_from_fchk('Number of contracted shells', self.path)[-1])

        self._atm = make_atm(self)
        self._bas, self._env = make_bas_env(self)

        self._basis = make_basis(self)
        self.nbas  = len(self._bas)

        process_basis(self)

        self.mf = MeanField2(path, self)
        self.iatsh = self.mf.iatsh
        self.mssh = self.mf.mssh
        self.mnsh = self.mf.mnsh
        self.expsh = self.mf.expsh
        self.c1 = self.mf.c1
        self.c2 = self.mf.c2 if hasattr(self.mf, 'c2') and self.mf.c2 is not None else None
        self.ncshell = self.mf.ncshell
        self.mo_coeff = self.mf.mo_coeff
        self.mo_occ = self.mf.mo_occ
        self.processed = True

    def build(self, self_consistent=None, verbose=None, atom=None, basis=None, **kwargs):
        if atom is not None:
            self.atom = atom

        if basis is not None:
            if basis == "minao":
                norm = 0.52917721092

                pyscf_mol = Mole2(self.path)
                pyscf_mol.atom=[(self.atomic_symbols[i], self.atom_coords()[i] * norm) for i in range(self.natoms)],
                pyscf_mol.basis="minao",
                pyscf_mol.spin=self.spin,
                pyscf_mol.charge=self.charge,
                pyscf_mol.cart=self.cart,
                pyscf_mol.symmetry=self.symmetry,

            else:
                self.basis = basis

        # Rebuild core GTO fields
        self._atm = make_atm(self)
        self._bas, self._env = make_bas_env(self)
        self._basis = make_basis(self)

        return self

    def _add_suffix(self, intor, cart=None):
        if not (intor[:4] == 'cint' or
                intor.endswith(('_sph', '_cart', '_spinor', '_ssc'))):
            if cart is None:
                cart = self.cart
            if cart:
                intor = intor + '_cart'
            else:
                intor = intor + '_sph'
        return intor

    def atom_symbol(self, pos):
        """Return the symbol of the atom at position pos."""
        if pos < 0 or pos >= self.natoms:
            raise IndexError("Atom index out of range.")
        return self.atomic_symbols[pos]

    def atom_symbols(self):
        """Return a list of atomic symbols for all atoms."""
        return self.atomic_symbols

    def make_atom(self):
        coords = self.atom_coords()  # shape (natoms, 3)
        return [(sym, tuple(coord)) for sym, coord in zip(self.atomic_symbols, coords)]

    def copy(self):
        from copy import deepcopy
        return deepcopy(self)

    def aoslice_by_atom(self, ao_loc=None):
        """Return AO slices per atom as (shell_start, shell_stop, ao_start, ao_stop)."""
        if ao_loc is None:
            ao_loc = self.ao_loc_nr()
        bas = self._bas
        natm = self.natoms
        slices = []
        shell_start = 0
        ao_start = 0
        for atm_id in range(natm):
            # Find shell_start for this atom
            while shell_start < len(bas) and bas[shell_start, 0] != atm_id:
                shell_start += 1
            shell_stop = shell_start
            while shell_stop < len(bas) and bas[shell_stop, 0] == atm_id:
                shell_stop += 1
            ao_start = ao_loc[shell_start]
            ao_stop = ao_loc[shell_stop]
            slices.append([shell_start, shell_stop, ao_start, ao_stop])
            shell_start = shell_stop
        return np.array(slices, dtype=np.int32)

    def atom_coords(self):
        coords = read_list_from_fchk('Current cartesian coordinates', self.path)
        return np.array(coords).reshape(int(self.natoms), 3)

    def intor_symmetric(self,str):
        if str == 'int1e_ovlp':
            if self._ovlp is not None:
                return self._ovlp
            print("Building overlap matrix...")
            from time import time
            start = time()
            process_basis(self)
            self._ovlp = build_ovlp(self)
            print("Overlap matrix built in {:.2f} seconds.".format(time() - start))
            return self._ovlp
        else:
            raise ValueError("Invalid integral type: {}".format(str))

    def atom_nelec_core(self, atm_id):
        '''Number of core electrons for pseudo potential.'''
        nuclear_charges = read_list_from_fchk('Nuclear charges', self.path)
        core_electrons = [self.atomic_numbers[i] - int(nuclear_charges[i]) for i in range(self.natoms)]
        return core_electrons[atm_id]

    def ao_loc_nr(self):
        bas = np.asarray(self._bas, dtype=np.int32)
        ANG_OF = 1
        NCTR_OF = 3

        def make_loc(bas, key):
            ang = bas[:, ANG_OF].astype(int)
            nctr = bas[:, NCTR_OF].astype(int)
            if 'cart' in key:
                dims = (ang + 1) * (ang + 2) // 2 * nctr
            elif 'sph' in key:
                dims = (2 * ang + 1) * nctr
            else:
                # fallback: treat as spherical
                dims = (2 * ang + 1) * nctr
            ao_loc = np.empty(len(dims) + 1, dtype=np.int32)
            ao_loc[0] = 0
            dims.cumsum(dtype=np.int32, out=ao_loc[1:])
            return ao_loc

        key = 'cart' if self.cart else 'sph'
        return make_loc(bas, key)

    def nao_nr(self):
        '''Number of basis functions.
        '''
        return self.nbasis

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

    def bas_nprim(self, atm_id):
        '''Number of primitive functions for each basis function.
        '''
        return int(self._bas[atm_id, 2])

    def bas_exp(self, atm_id):
        '''Exponent pointer for each basis function.
        '''
        return int(self._bas[atm_id, 5])

    def bas_ctr_coeff(self, atm_id):
        '''Contraction coefficient pointer for each basis function.
        '''
        return int(self._bas[atm_id, 6])

    def atom_pure_symbol(self, atm_id):
        '''Atom symbol for each basis function.
        '''
        return self.atomic_symbols[self.bas_atom(atm_id)]

    def has_ecp(self):
        '''Check if the molecule has ECP.
        '''
        nuclear_charges = read_list_from_fchk('Nuclear charges', self.path)
        nuclear_charges = [int(charge) for charge in nuclear_charges]
        return nuclear_charges != self.atomic_numbers

class MeanField2:
    def __init__(self, path, mol):
        self.path = path
        self.mol = mol
        self._processed = False
        self._reorder = True
        self.nao = self.mol.nao
        self.natm = self.mol.natm
        self.cart = self.mol.cart
        self.nelec = self.mol.nelec
        self.nelectron = self.nelec
        self._ovlp = None # Cache
        self._read_fchk = True
        self.nalpha = self.mol.nalpha
        self.__class__.__name__ = read_level_theory(self.path)[-2]
        if self.nalpha != self.mol.nbeta:
            wf = "unrest"
        else:
            wf = "rest"
        if wf == "rest":
            if 'HF' in self.__class__.__name__:
                self.__class__.__name__ = 'RHF'
            else:
                self.__class__.__name__ = 'RKS'
            self.orbital_energies = read_list_from_fchk('Alpha Orbital Energies', self.path)
            nocc = 0
            for num in self.orbital_energies:
                if num > 0:
                    nocc += 1
            self.mo_occ = np.array([2.] * (int(self.mol.nalpha + self.mol.nbeta) // 2) + [0.] * (len(self.orbital_energies) - (int(self.mol.nalpha+self.mol.nbeta) // 2)))
        elif wf == "unrest":
            if 'HF' in self.__class__.__name__:
                self.__class__.__name__ = 'UHF'
            else:
                self.__class__.__name__ = 'UKS'
            self.alpha_orbital_energies = read_list_from_fchk('Alpha Orbital Energies', self.path)
            self.beta_orbital_energies = read_list_from_fchk('Beta Orbital Energies', self.path)
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
        self.natm = self.natoms
        self.nbasis = int(read_from_fchk('Number of basis functions', self.path)[-1])
        self.numprim = int(read_from_fchk('Number of primitive shells', self.path)[-1])
        self.atomic_numbers = [int(i) for i in read_list_from_fchk('Atomic numbers', self.path)]
        self.atomic_symbols = read_atomic_symbols(self.atomic_numbers)
        self.numao = int(read_from_fchk('Number of basis functions', self.path)[-1])
        self.nummo = int(read_from_fchk('Number of independent functions', self.path)[-1])
        # Number of contracted shells
        self.ncshell = int(read_from_fchk('Number of contracted shells', self.path)[-1])

        # Number of primitive shells
        self.npshell = read_from_fchk('Number of primitive shells', self.path)[-1]
        # 0=s, 1=p, -1=sp, 2=6d, -2=5d, 3=10f, -3=7f
        # Shell types
        self.mssh = read_list_from_fchk('Shell types', self.path)
        self.mssh = [int(i) for i in self.mssh]
        # Number of primitives per shell
        self.mnsh = read_list_from_fchk('Number of primitives per shell', self.path)
        self.mnsh = [int(i) for i in self.mnsh]
        # Shell to atom map
        self.iatsh = read_list_from_fchk('Shell to atom map', self.path)
        self.iatsh = [int(i) for i in self.iatsh]
        # Primitive exponents
        self.expsh = read_list_from_fchk('Primitive exponents', self.path)
        # Current cartesian coordinates
        coords = read_list_from_fchk('Current cartesian coordinates', self.path)
        self.coord = np.array(coords).reshape(int(self.natoms), 3)

        with open(path, 'r') as file:
            if 'P(S=P)' in file.read():
                self.c1 = read_list_from_fchk('Contraction coefficients', self.path)
                self.c2 = read_list_from_fchk('P(S=P) Contraction coefficients', self.path)
            else:
                self.c1 = read_list_from_fchk('Contraction coefficients', self.path)
                c1s = []
                for c in range(len(self.c1)):
                    c1s.append((2*self.expsh[c]/np.pi)**(0.75) * self.c1[c])
                self.c1 = c1s
                self.c2 = None

        if wf == 'rest':
            self.mo_coeff = list(read_list_from_fchk('Alpha MO coefficients', self.path))
            self.mo_coeff = np.array(self.mo_coeff).reshape(self.nummo, self.numao).T

        elif wf == 'unrest':
            self.mo_coeff_alpha = read_list_from_fchk('Alpha MO coefficients',self.path)
            self.mo_coeff_beta = read_list_from_fchk('Beta MO coefficients', self.path)
            self.mo_coeff_alpha = np.array(self.mo_coeff_alpha).reshape(self.nummo, self.numao).T
            self.mo_coeff_beta = np.array(self.mo_coeff_beta).reshape(self.nummo, self.numao).T
            self.mo_coeff = [self.mo_coeff_alpha, self.mo_coeff_beta]

    def make_rdm1(self):
        """Computes density matrix."""
        if len(np.shape(self.mo_coeff)) == 2:  # RHF/RKS
            # Ensure mo_occ is correctly sized and used for slicing
            occupied_indices = np.where(self.mo_occ > 0)[0]
            mo_occ_coeffs = self.mo_coeff[:, occupied_indices]
            return np.dot(mo_occ_coeffs, mo_occ_coeffs.T) * 2
        elif len(np.shape(self.mo_coeff)) == 3:  # UHF/UKS
            occupied_indices_a = np.where(self.mo_occ[0] > 0)[0]
            occupied_indices_b = np.where(self.mo_occ[1] > 0)[0]
            mo_a_occ = self.mo_coeff[0][:, occupied_indices_a]
            mo_b_occ = self.mo_coeff[1][:, occupied_indices_b]
            rdm1_a = np.dot(mo_a_occ, mo_a_occ.T)
            rdm1_b = np.dot(mo_b_occ, mo_b_occ.T)
            return np.array([rdm1_a, rdm1_b])

    def aoslice_by_atom(self, ao_loc=None):
        return self.mol.aoslice_by_atom(ao_loc)

    def _add_suffix(self, intor, cart=None):
        if not (intor[:4] == 'cint' or
                intor.endswith(('_sph', '_cart', '_spinor', '_ssc'))):
            if cart is None:
                cart = self.cart
            if cart:
                intor = intor + '_cart'
            else:
                intor = intor + '_sph'
        return intor

    def get_ovlp(self):
        if hasattr(self.mol, '_ovlp') and self.mol._ovlp is not None:
            return self.mol._ovlp
        use_minao = getattr(self, 'minao', False)
        if use_minao:
            print("Building overlap matrix for minimal basis...")
        else:
            print("Building overlap matrix...")
        from time import time
        start = time()
        # Use Overlapper to compute and cache the overlap
        ovlp = Overlapper(self).get()
        self._ovlp = ovlp
        print("Overlap matrix built in {:.2f} seconds.".format(time() - start))
        return ovlp


import os
import numpy as np
from pyscf.gto import basis

class MoleANO:
    """
    Lightweight spherical ANO basis loader with FCHK-style internal arrays.
    Internal data is unrolled to match Gaussian's segmented shell requirement.
    """

    def __init__(self, mol):
        self.mol = mol
        self._ano = True
        self.cart = False
        self.atom_symbols = mol.atomic_symbols
        self.atomic_numbers = mol.atomic_numbers
        self.natm = mol.natm
        self.coord = mol.atom_coords()
        self.spin = mol.spin
        self.charge = mol.charge
        self.natoms = self.natm

        # build FCHK-style arrays for mnsh, mssh, iatsh, nctr, c1, expsh
        (self.iatsh, self.mssh, self.mnsh, self.nctr,
         self.c1, self.c2, self.expsh) = self._build_fchk_style()
        self.c2 = None

        # number of contracted shells / AOs
        self.ncshell = len(self.iatsh)

        # --- IMPORTANT: do NOT set a single self.nprim here; make_bas_env uses self.mnsh ---
        # self.nprim = len(self.expsh)   # <-- REMOVE/DO NOT SET

        # Now build PySCF style _bas, _env using your existing function
        # (Assumes make_bas_env and make_atm are defined in your scope)

        self._bas, self._env = make_bas_env(self)
        self._atm = make_atm(self)

        # ensure types match expectations
        self.mnsh = np.asarray(self.mnsh, dtype=np.int32)
        self.mssh = np.asarray(self.mssh, dtype=np.int32)
        self.iatsh = np.asarray(self.iatsh, dtype=np.int32)
        self.nctr = np.asarray(self.nctr, dtype=np.int32)
        self.c1 = np.asarray(self.c1, dtype=float)
        self.expsh = np.asarray(self.expsh, dtype=float)

        # number of contracted shells / AOs
        self.ncshell = len(self.iatsh)
        self.numao = int(self.ao_loc_nr()[-1])
        self.nao = self.numao
        self.nummo = self.numao

        # compute overlap for sanity (optional)
        self._ovlp = build_ovlp(self)


    def _build_fchk_style(self):
        ano_path = os.path.join(os.path.dirname(basis.__file__), "ano.dat")

        iatsh = []
        mssh = []
        mnsh = []
        nctr = []
        c1 = []
        expsh = []
        c2 = None

        for ia, sym in enumerate(self.atom_symbols):
            # Load basis info for the atom
            for shell in basis.parse_nwchem.load(ano_path, sym):
                l = int(shell[0])
                prims = np.array([np.atleast_1d(p) for p in shell[1:]], float)
                exps = prims[:, 0]

                # Coefficients shape: (N_prims, N_contractions)
                coeff = prims[:, 1:] if prims.shape[1] > 1 else np.ones((len(prims), 1))
                coeff = np.array(coeff, dtype=float, copy=True)

                n_contractions = coeff.shape[1]

                # --- UNROLLING GENERAL CONTRACTION ---
                # Gaussian/FCHK treats general contractions (e.g. ANOs) as distinct shells
                # sharing the same l, rather than one shell with multiple columns.
                for cid in range(n_contractions):
                    iatsh.append(int(ia + 1))  # FCHK is 1-based
                    mssh.append(int(l))
                    mnsh.append(int(len(prims)))

                    # In Gaussian segmented style, nctr is 1 per split shell
                    nctr.append(1)

                    # Extract the specific column for this contraction
                    c_col = coeff[:, cid]
                    c1.extend(c_col.flatten(order='C').tolist())

                    # Exponents must be duplicated for each unrolled shell
                    # to maintain alignment with mnsh in the flat arrays
                    expsh.extend(exps.tolist())

        return (
            np.array(iatsh, int),
            np.array(mssh, int),
            np.array(mnsh, int),
            np.array(nctr, int),
            np.array(c1, float),
            c2,
            np.array(expsh, float)
        )

    def ao_loc_nr(self):
        ao_loc = [0]
        for i in range(self.ncshell):
            l = abs(self.mssh[i])
            # nctr is now always 1 because we unrolled the general contractions
            n_contr = int(self.nctr[i])
            deg = 2 * l + 1
            ao_loc.append(ao_loc[-1] + deg * n_contr)
        return np.array(ao_loc, dtype=np.int32)

    def atom_coords(self):
        return self.mol.atom_coords()

    def nao_nr(self):
        return self.numao

    def aoslice_by_atom(self, ao_loc=None):
        """Return AO slices per atom as (shell_start, shell_stop, ao_start, ao_stop)."""
        if ao_loc is None:
            ao_loc = self.ao_loc_nr()
        bas = self._bas
        natm = self.natoms
        slices = []
        shell_start = 0
        ao_start = 0
        for atm_id in range(natm):
            # Find shell_start for this atom
            while shell_start < len(bas) and bas[shell_start, 0] != atm_id:
                shell_start += 1
            shell_stop = shell_start
            while shell_stop < len(bas) and bas[shell_stop, 0] == atm_id:
                shell_stop += 1
            ao_start = ao_loc[shell_start]
            ao_stop = ao_loc[shell_stop]
            slices.append([shell_start, shell_stop, ao_start, ao_stop])
            shell_start = shell_stop
        return np.array(slices, dtype=np.int32)

class MeanFieldMINAO():
    def __init__(self,  mol):
        # 2) keep a reference to the molecule
        self.mol  = mol
        self.atom_symbols = self.mol.atomic_symbols
        self.natoms = self.mol.natoms
        self.natm = self.mol.natm
        self.cart = False
        self.minao = True
        self.coord = self.mol.coord
        self.atomic_numbers = self.mol.atomic_numbers
        self.atomic_symbols = self.mol.atomic_symbols
        self.mo_coeff = self.mol.mo_coeff
        self.mo_occ = self.mol.mo_occ
        self.nelec = self.mol.nelec
        self.nelectron = self.nelec
        self.nalpha = self.mol.nalpha
        self.nbeta = self.mol.nalpha
        self.spin = 0
        self.charge = 0

        # 4) load the MINAO data into your SCF object
        self._load_minao()
        nao = sum((2 * int(row[1]) + 1) * int(row[3]) for row in self._bas)
        print("nao =", nao)

        # 5) now you can dynamically rename your class if you want
        #    (but only AFTER calling the RHF ctor!)
        self.__class__.__name__ = "RHF_MINAO"

        # 6) any other MeanField2-style bookkeeping:
        self._ovlp   = None
        process_basis(self)
        self._processed = True
        self._atm = make_atm(self)
        self._basis = make_basis(self)
        # number of shells/basis functions
        self.nbas = len(self._bas)
        # compute overlap on the MINAO basis (this will now use self._bas/_env/_atm)
        self._ovlp = build_ovlp(self)
        self.nao = int(self._ovlp.shape[0])

    def atom_symbols(self):
        return self.mol.atomic_symbols

    def atom_coords(self):
        return self.mol.atom_coords()

    def aoslice_by_atom(self, ao_loc=None):
        """Return AO slices per atom as (shell_start, shell_stop, ao_start, ao_stop)."""
        if ao_loc is None:
            ao_loc = self.ao_loc_nr()

        slices = []
        shell_idx = 0

        for atm_id in range(self.natoms):
            shell_start = shell_idx

            # Count number of shells for this atom
            # Note: self.iatsh stores 1-based atom indices; compare against atm_id+1
            while shell_idx < len(self.iatsh) and self.iatsh[shell_idx] == atm_id + 1:
                shell_idx += 1

            shell_stop = shell_idx

            ao_start = ao_loc[shell_start]
            ao_stop = ao_loc[shell_stop]

            slices.append((shell_start, shell_stop, ao_start, ao_stop))

        return np.array(slices, dtype=np.int32)

    def ao_loc_nr(self):
        bas = np.asarray(self._bas, dtype=np.int32)
        ANG_OF = 1
        NCTR_OF = 3

        def make_loc(bas, key):
            if "cart" in key:
                l = bas[:, ANG_OF].astype(int)
                dims = (l + 1) * (l + 2) // 2 * bas[:, NCTR_OF].astype(int)
            elif "sph" in key:
                dims = (bas[:, ANG_OF] * 2 + 1) * bas[:,NCTR_OF].astype(int)
            ao_loc = np.empty(len(dims) + 1, dtype=np.int32)
            ao_loc[0] = 0
            dims.cumsum(dtype=np.int32, out=ao_loc[1:])
            return ao_loc

        key = 'cart' if self.cart else 'sph'
        return make_loc(bas, key)

    def _load_minao(self):
        """Load the MINAO basis in PySCF format, building _bas/_env and
        auxiliary lists (iatsh, mssh, mnsh, nctr, expsh, c1).

        This implementation avoids duplicating identical basis blocks in
        _env: if the same element/l/exp/nctr block was already stored for
        a previous atom, it reuses the ptr_exp/ptr_coeff pointers.
        """
        from pyscf.gto.basis import minao
        from pyscf.gto.mole import gto_norm, _nomalize_contracted_ao

        env = [0.0] * 20
        for x, y, z in self.atom_coords():
            env.extend([x, y, z, 0.0])

        bas = []
        iatsh = []
        mssh = []
        mnsh = []
        nctr = []
        expsh = []
        c1 = []

        env_idx = len(env)

        # Reuse map for already-added basis blocks: (sym,l,exps_tuple,nctr) -> (ptr_exp, ptr_coeff)
        basis_block_map = {}

        for ia, symb in enumerate(self.atomic_symbols):
            try:
                basis = getattr(minao, symb)
            except AttributeError:
                raise ValueError(f"MINAO does not contain entry for {symb}")

            for shell in basis:
                l = int(shell[0])
                prims = np.array([np.atleast_1d(p) for p in shell[1:]], dtype=float)
                exponents = prims[:, 0]
                raw_coeffs = prims[:, 1:] if prims.shape[1] > 1 else np.ones((len(exponents), 1))

                # normalize primitives (as PySCF does)
                coeffs = raw_coeffs * gto_norm(l, exponents)[:, None]
                #coeffs = _nomalize_contracted_ao(l, exponents, coeffs)

                nprim = int(len(exponents))
                nctr_shell = int(coeffs.shape[1])

                exp_key = tuple(np.round(exponents, 8).tolist())
                map_key = (symb, int(l), exp_key, nctr_shell)
                existing = basis_block_map.get(map_key)
                if existing is not None:
                    ptr_exp, ptr_coeff = existing
                else:
                    ptr_exp = env_idx
                    env.extend(exponents.tolist())
                    env_idx += nprim

                    ptr_coeff = env_idx
                    # store coefficients in row-major order (nprim rows × nctr cols)
                    env.extend(coeffs.flatten(order='F').tolist())
                    env_idx += nprim * nctr_shell

                    basis_block_map[map_key] = (ptr_exp, ptr_coeff)

                bas.append([ia, int(l), nprim, nctr_shell, 0, ptr_exp, ptr_coeff, 0])

                # auxiliary per-shell bookkeeping
                iatsh.append(ia + 1)
                mssh.append(int(l))
                mnsh.append(nprim)
                nctr.append(nctr_shell)
                expsh.extend(exponents.tolist())
                c1.extend(coeffs.flatten(order='F').tolist())

        # store
        self._bas = np.array(bas, dtype=np.int32)
        self._env = np.array(env, dtype=np.float64)

        self.iatsh = iatsh
        self.mssh = mssh
        self.mnsh = mnsh
        self.nctr = nctr
        self.expsh = expsh
        self.c1 = c1
        self.ncshell = len(self.mssh)
        self.numprim = len(self.expsh)
        # spherical degens: 2*l + 1
        self.nbasis = int(sum(2 * int(l) + 1 for l in self.mssh))

def mult(shell_type):
    mult_dict = { #s=0, p=1, d=2, f=3, g=4, h=5, sp=-1, dcart=-2, fcart=-3, gcart=-4, hcart=-5
         0: 1,  1: 3,  2: 6,  3: 10, 4: 15, 5: 21,
         - 5: 11, -4: 9, -3: 7, -2: 5, -1: 4,
    }
    return mult_dict.get(shell_type, 0)

def process_basis(mf):
    mssh_abs = np.abs(mf.mssh)
    mult_vals = np.array([mult(s) for s in mf.mssh])
    mult_abs_vals = np.array([mult(s) for s in mssh_abs])

    # Total number of primitives and basis functions
    # Follow Fortran logic: if mssh < -1 use mult(abs(mssh)), else use mult(mssh)
    numprim = int(np.sum([ (mult(abs(s)) if s < -1 else mult(s)) * n for s, n in zip(mf.mssh, mf.mnsh) ]))
    nbasis = np.sum(mult_vals)

    # Basis to atom map
    ihold = np.repeat(mf.iatsh, mult_vals)

    # Basis set limits
    llim = [1]
    iulum = [0] * mf.natoms
    for iat in range(1, mf.natoms + 1):
        indices = np.where(ihold == iat)[0]
        if len(indices) > 0:
            iulum[iat - 1] = indices[-1] + 1
            llim.append(indices[-1] + 2)
    if iulum[-1] == 0:
        iulum[-1] = nbasis

    # expp and coefp arrays
    expp, coefp = [], []
    jcount = 0
    for shell_type, nprim in zip(mf.mssh, mf.mnsh):
        # For SP shells, number of primitives per shell (kk) is 4
        if shell_type == -1:
            kk = 4
        else:
            kk = mult(abs(shell_type))

        for _ in range(nprim):
            exp_val = float(mf.expsh[jcount])
            c1val = float(mf.c1[jcount])
            c2val = float(mf.c2[jcount]) if (hasattr(mf, 'c2') and mf.c2 is not None) else 0.0
            for k in range(1, kk + 1):
                expp.append(exp_val)
                # SP shells use c1 for k==1, c2 otherwise
                if shell_type == -1 and k != 1:
                    coefp.append(c2val)
                else:
                    coefp.append(c1val)
            jcount += 1

    expp = np.array(expp, dtype=float)
    coefp = np.array(coefp, dtype=float)

    # NLM and normalization
    nlm, iptoat = get_nlm(mf)
    nn, ll, mm = nlm[:, 0], nlm[:, 1], nlm[:, 2]
    fnn = factorial(nn) / factorial(2 * nn)
    fll = factorial(ll) / factorial(2 * ll)
    fmm = factorial(mm) / factorial(2 * mm)
    xnorm = (2.0 * expp / np.pi) ** 0.75 * np.sqrt((8.0 * expp) ** (nn + ll + mm) * fnn * fll * fmm)

    mf.coefp = coefp
    mf.xnorm = xnorm
    mf.nbasis = nbasis
    mf.numprim = numprim
    coefpb, iptob_cartesian = prim_orb_mapping(mf)

    # Maximum primitives per basis function
    mmax = np.max([
        3 * n if s <= -2 else (2 * n if s == -4 else n)
        for s, n in zip(mf.mssh, mf.mnsh)
    ]) + 1

    # Primitive-to-basis mapping
    nprimbas = np.zeros((mmax, nbasis), dtype=int)
    for i in range(nbasis):
        prim_indices = np.where(np.abs(coefpb[:, i]) > 1.0e-10)[0] + 1
        nprimbas[:len(prim_indices), i] = prim_indices

    # Updating the mf object
    mf.ihold = ihold.tolist()
    mf.llim = llim
    mf.iulum = iulum
    mf.nlm = nlm
    mf.iptoat = iptoat
    mf.coefpb = coefpb
    mf.iptob_cartesian = iptob_cartesian
    mf.mmax = mmax
    mf.nprimbas = nprimbas
    mf.expp = expp
    mf.numprim = len(expp)

import numpy as np
from scipy.special import factorial  # vectorized factorial

import numpy as np
from pyscf.gto.moleintor import getints

MAPS = {
    -2: [4, 2, 0, 1, 3],  # 5D spherical (5 funcs)
     2: [0, 3, 4, 1, 5, 2],  # 6D cartesian (6 funcs)
    -3: [6, 4, 2, 0, 1, 3, 5],  # 7F spherical (7 funcs)
     3: [0, 4, 5, 1, 6, 2, 7, 3, 8, 9],  # 10F cartesian (10 funcs)
    -4: [8, 6, 4, 2, 0, 1, 3, 5, 7],  # 9G spherical (9 funcs)
     4: [0, 6, 7, 1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 13, 14],  # 15G cartesian (15 funcs)
}


def build_ovlp(mf):
    """
    Builds the overlap matrix for the given molecule and reorders the AO
    basis functions inside each shell according to MAPS so the AO ordering
    matches the expected (spherical/cart) convention.

    The reordering is applied *after* getints(...) returns the overlap matrix.
    """
    # Ensure basis control variables exist
    if not getattr(mf, '_processed', False):
        process_basis(mf)
        mf._processed = True

    # MINAO caching
    if hasattr(mf, 'minao') and mf.minao:
        if hasattr(mf, '_ovlp_minao') and mf._ovlp_minao is not None:
            return mf._ovlp_minao
    else:
        if hasattr(mf, '_ovlp') and mf._ovlp is not None:
            return mf._ovlp

    # Use mf's own _bas/_env/_atm if they exist (for MoleANO etc), otherwise get from mf.mol
    if hasattr(mf, "minao") and mf.minao:
        _bas = np.asarray(mf._bas, dtype=np.int32)
        _env = np.asarray(mf._env, dtype=np.float64)
        _atm = np.asarray(mf._atm, dtype=np.int32)
    elif hasattr(mf, '_bas') and mf._bas is not None and hasattr(mf, 'mol'):
        _bas = np.asarray(mf._bas, dtype=np.int32)
        _env = np.asarray(mf._env, dtype=np.float64)
        _atm = np.asarray(mf._atm, dtype=np.int32)
    elif hasattr(mf, 'mol'):
        _bas = np.asarray(mf.mol._bas, dtype=np.int32)
        _env = np.asarray(mf.mol._env, dtype=np.float64)
        _atm = np.asarray(mf.mol._atm, dtype=np.int32)
    else:
        _bas = np.asarray(mf._bas, dtype=np.int32)
        _env = np.asarray(mf._env, dtype=np.float64)
        _atm = np.asarray(mf._atm, dtype=np.int32)

    mf._bas = _bas
    mf._env = _env
    mf._atm = _atm
    # pick correct intor suffix to match cart/sph convention
    mol_for_suffix = mf if not hasattr(mf, 'mol') else mf.mol
    intor = mol_for_suffix._add_suffix("int1e_ovlp", cart=getattr(mf, 'cart', None))

    # compute raw overlap
    from pyscf.gto.moleintor import getints
    s = getints(intor, _atm, _bas, _env, hermi=1)
    mf._s_no_order = s.copy()

    if getattr(mf, "_reorder", True):
        from esipy.tools import permute_aos
        s = permute_aos(s, mf)

    # store and return
    if hasattr(mf, 'minao') and getattr(mf, 'minao'):
        mf._ovlp_minao = s
    else:
        mf._ovlp = s
    return s

    # Convert to NumPy arrays
    numprim, nbasis = mf.numprim, mf.nbasis
    coord = np.asarray(mf.coord)
    iptoat = np.asarray(mf.iptoat) - 1  # make 0-based
    nlm = np.asarray(mf.nlm, dtype=int)
    expp = np.asarray(mf.expp)
    coefpb = np.asarray(mf.coefpb)
    # Precompute factorials for all l values
    max_l = int(np.max(nlm))
    max_fact_arg = 2 * max_l;
    fact_table = factorial(np.arange(max_fact_arg + 1), exact=True)

    def fact(n):  # fast lookup
        return fact_table[n]

    # --- Part 1: primitive overlap matrix (sp) ---
    coords_p = coord[iptoat]
    AminusB = coords_p[:, None, :] - coords_p[None, :, :]
    AminusB2 = AminusB**2

    expp_ia = expp[:, None]
    expp_ib = expp[None, :]
    gamma_p = expp_ia + expp_ib
    eta_p = (expp_ia * expp_ib) / gamma_p
    sp_base = (np.pi / gamma_p)**1.5

    sp = np.zeros((numprim, numprim))
    idx_ia, idx_ib = np.triu_indices(numprim)

    for ia, ib in zip(idx_ia, idx_ib):
        AmB = AminusB[ia, ib]
        AmB2_pair = AminusB2[ia, ib]
        etap_val = eta_p[ia, ib]
        exp_ia_val = expp[ia]
        exp_ib_val = expp[ib]
        l1_vec = nlm[ia]
        l2_vec = nlm[ib]

        # Zero overlap check
        tol = 1.0e-14
        if any(abs(AmB[ax]) < tol and (l1_vec[ax] + l2_vec[ax]) % 2 != 0 for ax in range(3)):
            continue

        angular_factor_product = 1.0
        for ax in range(3):
            l1 = l1_vec[ax]
            l2 = l2_vec[ax]
            AmB_ax = AmB[ax]

            term_xyz = fact(l1) * fact(l2) / (2.0 ** (l1 + l2))
            if abs(AmB_ax) > tol:
                term_xyz *= np.exp(-etap_val * AmB2_pair[ax])

            sum_i = 0.0
            for i1 in range(l1 // 2 + 1):
                j1 = l1 - 2 * i1
                for i2 in range(l2 // 2 + 1):
                    j2 = l2 - 2 * i2
                    j = j1 + j2

                    if abs(AmB_ax) > tol:
                        sum_r = 0.0
                        for ir in range(j // 2 + 1):
                            den = fact(ir) * fact(j - 2 * ir)
                            if den == 0:
                                continue
                            num = (2.0 * AmB_ax) ** (j - 2 * ir)
                            xfac = (etap_val ** (j - ir)) * num / den
                            if ir % 2 != 0:
                                xfac = -xfac
                            sum_r += xfac
                    elif j % 2 == 0:
                        k_idx = j // 2
                        sum_r = (etap_val ** k_idx) / fact(k_idx)
                        if k_idx % 2 != 0:
                            sum_r = -sum_r
                    else:
                        sum_r = 0.0

                    if j1 % 2 != 0:
                        sum_r = -sum_r

                    den_fac = fact(i1) * fact(j1) * fact(i2) * fact(j2)
                    if den_fac != 0:
                        sum_i += sum_r * fact(j) / (
                            den_fac * (exp_ia_val ** (l1 - i1)) * (exp_ib_val ** (l2 - i2))
                        )
            angular_factor_product *= sum_i * term_xyz

        sp[ia, ib] = sp_base[ia, ib] * angular_factor_product

    # Symmetrize
    sp = sp + sp.T - np.diag(np.diag(sp))

    # --- Part 2: contraction to basis ---
    s = coefpb.T @ sp @ coefpb
    if hasattr(mf, 'minao') and mf.minao:
        mf._ovlp_minao = s
    else:
        mf._ovlp = s
    return s

from pyscf.gto import moleintor
from pyscf.gto.mole import conc_env

def build_cross_ovlp(mf1, mf2, tol=1e-8):
    """Compute cross overlap block between bases of mf1 and mf2.

    mf1/mf2 may be either Mole-like objects or wrappers with .mol attribute
    (e.g. mean-field objects). Returns a 2-D ndarray with shape (nao1, nao2).
    """

    from copy import deepcopy
    from esipy.tools import permute_aos_cross
    mol1 = deepcopy(mf1)
    mol2 = deepcopy(mf2)
    np.set_printoptions(precision=4, suppress=True, threshold=np.inf)
    bas1, atm1, env1 = mol1._bas, mol1._atm, mol1._env
    bas2, atm2, env2 = mol2._bas, mol2._atm, mol2._env
    nbas1 = len(bas1)
    nbas2 = len(bas2)

    # build combined environment as PySCF does
    atmc, basc, envc = conc_env(atm1, bas1, env1, atm2, bas2, env2)

    shls_slice = (0, nbas1, nbas1, nbas1 + nbas2)
    # decide which integral string to call
    base_intor = 'int1e_ovlp'
    cart1 = getattr(mol1, 'cart', False)
    cart2 = getattr(mol2, 'cart', False)

    # If both agree on cart/sph, call that suffix directly

    if cart1 == cart2:
        intor = base_intor + ('_cart' if cart1 else '_sph')
        out = moleintor.getints(intor, atmc, basc, envc, shls_slice, None, 0)
        out = permute_aos_cross(out, mol1, mol2)
        return out

    # Mixed case: compute cart integrals and convert
    # First compute cart version
    intor_cart = base_intor + '_cart'
    mat_cart = moleintor.getints(intor_cart, atmc, basc, envc, shls_slice, None, 0)

    # If mol2 is spherical but mol1 is cart: convert columns (mol2) cart->sph
    if cart1 and not cart2:
        # need mol2.cart2sph_coeff() returning matrix shape (ncart_ao2, nsph_ao2)
        if hasattr(mol2, 'cart2sph_coeff'):
            coeff2 = mol2.cart2sph_coeff()
            # mat_cart has shape (nao1_cart, nao2_cart); columns -> transform by coeff2
            return mat_cart.dot(coeff2)
        else:
            raise RuntimeError("build_cross_ovlp: mol2 lacks cart2sph_coeff(); cannot convert cart->sph")

    # If mol2 is cart and mol1 spherical: convert rows (mol1) cart->sph
    if cart2 and not cart1:
        if hasattr(mol1, 'cart2sph_coeff'):
            coeff1 = mol1.cart2sph_coeff()
            # coeff1 converts cart->sph for mol1: do coeff1.T @ mat_cart
            return coeff1.T.dot(mat_cart)
        else:
            raise RuntimeError("build_cross_ovlp: mol1 lacks cart2sph_coeff(); cannot convert cart->sph")

    # fallback (shouldn't be reached)
    raise RuntimeError("Unhandled cart/sph combination in build_cross_ovlp")

    # --- Python slow implementation to compute the cross-overlap matrix ---
    # Convert to NumPy arrays
    coord1 = np.asarray(mf1.coord)
    coord2 = np.asarray(mf2.coord)
    iptoat1 = np.asarray(mf1.iptoat) - 1
    iptoat2 = np.asarray(mf2.iptoat) - 1
    nlm1 = np.asarray(mf1.nlm, dtype=int)
    nlm2 = np.asarray(mf2.nlm, dtype=int)
    expp1 = np.asarray(mf1.expp)
    expp2 = np.asarray(mf2.expp)
    coefpb1 = np.asarray(mf1.coefpb)
    coefpb2 = np.asarray(mf2.coefpb)

    # Precompute factorials
    max_l = max(np.max(nlm1), np.max(nlm2))
    max_fact_arg = 2 * max_l
    fact_table = factorial(np.arange(max_fact_arg + 1), exact=True)
    def fact(n):
        return fact_table[n]

    # Primitive centers and differences
    coords_p1 = coord1[iptoat1]
    coords_p2 = coord2[iptoat2]
    AminusB = coords_p1[:, None, :] - coords_p2[None, :, :]
    AminusB2 = AminusB**2

    expp_ia = expp1[:, None]
    expp_ib = expp2[None, :]
    gamma_p = expp_ia + expp_ib
    eta_p = (expp_ia * expp_ib) / gamma_p
    sp_base = (np.pi / gamma_p)**1.5

    sp = np.zeros((numprim1, numprim2))

    # --- core loop: identical math to build_ovlp ---
    for ia in range(numprim1):
        l1_vec = nlm1[ia]
        exp_ia_val = expp1[ia]
        for ib in range(numprim2):
            l2_vec = nlm2[ib]
            exp_ib_val = expp2[ib]
            AmB = AminusB[ia, ib]
            AmB2_pair = AminusB2[ia, ib]
            etap_val = eta_p[ia, ib]

            # zero overlap check
            if any(abs(AmB[ax]) < tol and (l1_vec[ax] + l2_vec[ax]) % 2 != 0 for ax in range(3)):
                continue
            angular_factor_product = 1.0
            for ax in range(3):
                l1 = l1_vec[ax]
                l2 = l2_vec[ax]
                AmB_ax = AmB[ax]
                term_xyz = fact(l1) * fact(l2) / (2.0 ** (l1 + l2))
                if abs(AmB_ax) > tol:
                    term_xyz *= np.exp(-etap_val * AmB2_pair[ax])

                sum_i = 0.0
                for i1 in range(l1 // 2 + 1):
                    j1 = l1 - 2 * i1
                    for i2 in range(l2 // 2 + 1):
                        j2 = l2 - 2 * i2
                        j = j1 + j2

                        if abs(AmB_ax) > tol:
                            sum_r = 0.0
                            for ir in range(j // 2 + 1):
                                den = fact(ir) * fact(j - 2 * ir)
                                if den == 0:
                                    continue
                                num = (2.0 * AmB_ax) ** (j - 2 * ir)
                                xfac = (etap_val ** (j - ir)) * num / den
                                if ir % 2 != 0:
                                    xfac = -xfac
                                sum_r += xfac
                        elif j % 2 == 0:
                            k_idx = j // 2
                            sum_r = (etap_val ** k_idx) / fact(k_idx)
                            if k_idx % 2 != 0:
                                sum_r = -sum_r
                        else:
                            sum_r = 0.0

                        if j1 % 2 != 0:
                            sum_r = -sum_r

                        den_fac = fact(i1) * fact(j1) * fact(i2) * fact(j2)
                        if den_fac != 0:
                            sum_i += sum_r * fact(j) / (
                                den_fac * (exp_ia_val ** (l1 - i1)) * (exp_ib_val ** (l2 - i2))
                            )

                angular_factor_product *= sum_i * term_xyz

            sp[ia, ib] = sp_base[ia, ib] * angular_factor_product

    # contract
    s = coefpb1.T @ sp @ coefpb2

    # cache
    mf1._cross_ovlp_cache = s
    mf1._cross_ovlp_basis2 = mf2
    return s


def iao(mf_orig, mf_min, coeffs=None):
    """
    Build IAOs using a minimal basis (mf_min) and the original AO basis (mf_orig).

    mf_orig, mf_min:
        Objects containing:
        - nbasis, numprim
        - coefpb, expp, coord, iptoat, nlm
        - All data needed by build_cross_ovlp()

    mo_coeff:
        Molecular orbital coefficients in AO basis (nbas_orig × nmo)

    nelectron:
        Total number of electrons (for closed-shell, we take nocc = nelec//2)
    """

    # --- Step 1: Overlap matrices ---
    S1 = build_ovlp(mf_orig)  # AO-AO overlap (square, nbas_orig×nbas_orig)
    S12 = build_cross_ovlp(mf_orig, mf_min)  # AO-MINAO overlap (nbas_orig×nbas_min)
    S2  = build_ovlp(mf_min)  # MINAO-MINAO overlap (square, nbas_min×nbas_min)

    # --- Step 2: Occupied space ---
    nocc = mf_orig.nelec // 2
    if coeffs is None:
        coeffs = mf_min.mo_coeff  # (nbas_orig × nmo)
    C_occ = coeffs[:, :nocc]  # (nbas_orig × nocc)

    # --- Step 3: Build projectors ---
    S21 = S12.T
    Ctild_min = np.linalg.solve(S2, S21 @ C_occ)  # (nmin × nocc)

    try:
        P12 = np.linalg.solve(S1, S12)  # (nbas_orig × nbasis_min)
        Ctild_AO = np.linalg.solve(S1, S12 @ Ctild_min)
    except np.linalg.LinAlgError:
        # Fallback to canonical orthonormalization
        X = scf.addons.canonical_orth_(S1, lindep_threshold=1e-8)
        P12 = X @ X.T @ S12
        Ctild_AO = P12 @ Ctild_min

    # --- Step 4: Orthonormalize projected orbitals ---
    from pyscf.lo.orth import vec_lowdin
    Ctild = vec_lowdin(Ctild_AO, S1)

    # --- Step 5: PySCF-style IAO projector formula ---
    P_occ  = C_occ @ C_occ.T @ S1
    P_proj = Ctild @ Ctild.T @ S1
    IAOs = P12 + 2 * (P_occ @ P_proj @ P12) - P_occ @ P12 - P_proj @ P12

    # --- Step 6: Orthonormalize IAOs ---
    IAOs = vec_lowdin(IAOs, S1)

    return IAOs


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
    # Minimal, defensive normalization: ensure primitive arrays are 1D float arrays
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

