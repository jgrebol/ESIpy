import os
import numpy as np
from pyscf import gto, scf
from esipy.tools import permute_aos_rows

_readers = {}

class FchkReader:
    def __init__(self, path):
        self.path = path
        with open(path, 'r') as f: self.lines = [ln.rstrip('\n') for ln in f]
        self._index = {}
        for i, line in enumerate(self.lines):
            if len(line) >= 43 and line[43:45] in ['I ', 'R ', 'C ', 'L ']:
                label = line[:43].strip()
                if label not in self._index: self._index[label] = i
        self._single_cache = {}
        self._list_cache = {}

    def find_first(self, prefix):
        if prefix in self._single_cache: return self._single_cache[prefix]
        if prefix in self._index: line = self.lines[self._index[prefix]]; tokens = line.split(); self._single_cache[prefix] = tokens; return tokens
        for line in self.lines:
            if line.startswith(prefix): tokens = line.split(); self._single_cache[prefix] = tokens; return tokens
        return None

    def read_list(self, start, count=None):
        key = (start, count)
        if key in self._list_cache: return self._list_cache[key]
        start_idx = self._index.get(start)
        if start_idx is None:
            for i, line in enumerate(self.lines):
                if line.startswith(start): start_idx = i; break
        if start_idx is None: self._list_cache[key] = []; return []
        header = self.lines[start_idx]
        if count is None:
            import re
            m = re.search(r"(\d+)", header.split()[-1])
            if m: count = int(m.group(1))
        out, j = [], start_idx + 1
        while j < len(self.lines) and (count is None or len(out) < count):
            line = self.lines[j].strip()
            if not line: j += 1; continue
            for tok in line.split():
                try: out.append(float(tok))
                except: pass
                if count is not None and len(out) >= count: break
            j += 1
        self._list_cache[key] = out[:count] if count is not None else out
        return self._list_cache[key]

def _get_reader(path):
    path = os.path.abspath(path)
    if path in _readers: return _readers[path]
    r = FchkReader(path); _readers[path] = r; return r

def read_from_fchk(to_read, path): return _get_reader(path).find_first(to_read) or []
def read_list_from_fchk(start, path): return _get_reader(path).read_list(start)
def read_level_theory(path): r = _get_reader(path); return r.lines[1].split() if len(r.lines) >= 2 else []

def read_atomic_symbols(z):
    z_to_symbol = {1:'H',2:'He',3:'Li',4:'Be',5:'B',6:'C',7:'N',8:'O',9:'F',10:'Ne',11:'Na',12:'Mg',13:'Al',14:'Si',15:'P',16:'S',17:'Cl',18:'Ar',19:'K',20:'Ca',21:'Sc',22:'Ti',23:'V',24:'Cr',25:'Mn',26:'Fe',27:'Co',28:'Ni',29:'Cu',30:'Zn',31:'Ga',32:'Ge',33:'As',34:'Se',35:'Br',36:'Kr',37:'Rb',38:'Sr',39:'Y',40:'Zr',41:'Nb',42:'Mo',43:'Tc',44:'Ru',45:'Rh',46:'Pd',47:'Ag',48:'Cd',49:'In',50:'Sn',51:'Sb',52:'Te',53:'I',54:'Xe',55:'Cs',56:'Ba',57:'La',58:'Ce',59:'Pr',60:'Nd',61:'Pm',62:'Sm',63:'Eu',64:'Gd',65:'Tb',66:'Dy',67: 'Ho', 68: 'Er', 69: 'Tm', 70: 'Yb',71: 'Lu', 72: 'Hf', 73: 'Ta', 74: 'W', 75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt', 79: 'Au', 80: 'Hg',81: 'Tl', 82: 'Pb', 83: 'Bi', 84: 'Po', 85: 'At', 86: 'Rn', 87: 'Fr', 88: 'Ra', 89: 'Ac', 90: 'Th',91: 'Pa', 92: 'U', 93: 'Np', 94: 'Pu', 95: 'Am', 96: 'Cm', 97: 'Bk', 98: 'Cf', 99: 'Es', 100: 'Fm',101: 'Md', 102: 'No', 103: 'Lr', 104: 'Rf', 105: 'Db', 106: 'Sg', 107: 'Bh', 108: 'Hs', 109: 'Mt',110: 'Ds', 111: 'Rg', 112: 'Cn', 113: 'Nh', 114: 'Fl', 115: 'Mc', 116: 'Lv', 117: 'Ts', 118: 'Og'}
    return [z_to_symbol[int(i)] for i in z]

class Mole2:
    def __init__(self, path):
        self.path = path; self.fchk = FchkMolecule(path)
        self.atomic_numbers, self.atomic_symbols = self.fchk.atomic_numbers, self.fchk.atomic_symbols
        self.natm, self.charge = self.fchk.natoms, self.fchk.charge
        self.nalpha, self.nbeta = self.fchk.nalpha, self.fchk.nbeta
        self.spin, self.nelectron = self.fchk.mult - 1, self.fchk.nalpha + self.fchk.nbeta
        self.unrestricted, self.cart = self.fchk.unrestricted, self.fchk.cart
        self.nummo, self.numao = self.fchk.nummo, self.fchk.numao
        bt = read_level_theory(self.path); self.basis = bt[-1] if bt else None
        self.coord = self.fchk.coord
        self.fchk_basis_arrays = {'mssh': self.fchk.mssh, 'mnsh': self.fchk.mnsh, 'iatsh': self.fchk.iatsh, 'expsh': self.fchk.expsh, 'c1': self.fchk.c1, 'c2': self.fchk.c2, 'ncshell': self.fchk.ncshell}
        self._basis = make_basis(self); self.pyscf_mol = gto.Mole()
        self.pyscf_mol.atom = [(sym, tuple(xyz)) for sym, xyz in zip(self.atomic_symbols, self.coord)]
        self.pyscf_mol.basis = self._basis; self.pyscf_mol.charge = int(self.charge); self.pyscf_mol.spin = int(self.spin)
        self.pyscf_mol.cart = self.cart; self.pyscf_mol.unit = 'Bohr'; self.pyscf_mol.verbose = 0
        self.ecp = None; self.nuclear_charges = self.fchk.nuclear_charges

    def build(self, *args, **kwargs):
        kwargs.setdefault('verbose', 0)
        if self.ecp is None and self.nuclear_charges is not None:
             if not np.allclose(self.atomic_numbers, self.nuclear_charges): raise RuntimeError("Calculation requires ecp, but no ECP is defined")
        self.pyscf_mol.ecp = self.ecp
        if self.ecp:
            print(f" | Applying ECP: {self.ecp}")
        self.pyscf_mol.build(*args, **kwargs)
        self._bas, self._env, self._atm = self.pyscf_mol._bas, self.pyscf_mol._env, self.pyscf_mol._atm
        self.nao, self.nbas = int(self.pyscf_mol.nao_nr()), int(self.pyscf_mol.nbas)
        self.atom, self.nelec = self.pyscf_mol.atom, self.pyscf_mol.nelec
        return self

    def __getattr__(self, name): return getattr(self.pyscf_mol, name)

class MeanField2:
    def __init__(self, path, mole2: Mole2):
        self.path, self.mole2, self.mol = path, mole2, mole2.pyscf_mol
        self.nao, self.nummo = self.mole2.numao, self.mole2.nummo
        self.nalpha, self.nbeta = self.mole2.nalpha, self.mole2.nbeta
        self.e_tot, self.charge = float(getattr(self.mole2.fchk, 'e_tot', 0.0)), self.mole2.charge
        self.is_qchem = getattr(self.mole2.fchk, 'is_qchem', False)
        if self.is_qchem and any(term in os.path.basename(path).lower() for term in ['mp2', 'ccsd', 'cc', 'ci']):
            raise NotImplementedError("Correlated wavefunctions from Q-Chem FCHK files are not supported because Q-Chem does not write the correlated density to the FCHK file.")
        S_raw = self.mol.intor_symmetric('int1e_ovlp')
        v_align = 1.0 / np.sqrt(np.abs(np.diag(S_raw)))

        def standardize_mo(mo_arr): from esipy.tools import permute_aos_rows; return permute_aos_rows(mo_arr, self.mole2)
        def standardize_dm(dm_arr): from esipy.tools import permute_aos_rows; dm_p = permute_aos_rows(dm_arr, self.mole2); return permute_aos_rows(dm_p.T, self.mole2).T

        mo_a_flat = read_list_from_fchk('Alpha MO coefficients', path)
        mo_b_flat = read_list_from_fchk('Beta MO coefficients', path)
        is_uhf = (len(mo_b_flat) > 0) or self.mole2.unrestricted
        
        mo_read_success = False
        if is_uhf and len(mo_a_flat) > 0 and len(mo_b_flat) > 0:
            print(" | Using Canonical UHF MO coefficients")
            ma = np.array(mo_a_flat, dtype=float).reshape(self.nummo, self.nao).T
            mb = np.array(mo_b_flat, dtype=float).reshape(self.nummo, self.nao).T
            self.mo_coeff = [standardize_mo(ma), standardize_mo(mb)]
            self.mo_occ = [np.zeros(self.nummo), np.zeros(self.nummo)]
            self.mo_occ[0][:self.nalpha], self.mo_occ[1][:self.nbeta] = 1.0, 1.0
            mo_read_success, self._scf, self.__name__ = True, scf.UHF(self.mol), "UHF"
        
        if not mo_read_success:
            d_labels = [('Total CI Rho(1) Density', 'Spin CI Rho(1) Density'), ('Total CI Density', 'Spin CI Density'), ('Total CC Density', 'Spin CC Density'), ('Total MP2 Density', 'Spin MP2 Density'), ('Total SCF Density', 'Spin SCF Density')]
            found_density = False
            for t_lbl, s_lbl in d_labels:
                dt_flat = read_list_from_fchk(t_lbl, path)
                if len(dt_flat) > 0:
                    print(f" | Found density: {t_lbl}"); found_density = True; n = self.nao
                    if len(dt_flat) == n*(n+1)//2:
                        mat_gau = np.zeros((n, n)); mat_gau[np.tril_indices(n)] = dt_flat
                        mat_gau = mat_gau + mat_gau.T - np.diag(np.diag(mat_gau))
                    else: mat_gau = np.array(dt_flat).reshape(n, n)
                    dt = standardize_dm(mat_gau)
                    from scipy.linalg import eigh
                    occ, coeff = eigh(S_raw @ dt @ S_raw, b=S_raw)
                    idx = np.argsort(occ)[::-1]
                    self.mo_occ, self.mo_coeff = occ[idx], coeff[:, idx]
                    self.mo_occ[self.mo_occ < 1e-12] = 0.0
                    self._scf, self.__name__ = scf.RHF(self.mol), "RHF"; self._scf.make_rdm1 = lambda *args, **kwargs: dt
                    self._scf.density_label = t_lbl
                    break
            if not found_density:
                if len(mo_a_flat) > 0:
                    print(" | Using Fallback RHF/ROHF MO coefficients")
                    mo_arr_a = np.array(mo_a_flat, dtype=float).reshape(self.nummo, self.nao).T
                    self.mo_coeff = standardize_mo(mo_arr_a)
                    self.mo_occ = np.zeros(self.nummo); self.mo_occ[:(self.nalpha+self.nbeta)//2] = 2.0
                    self._scf, self.__name__ = scf.RHF(self.mol), "RHF"
                else: raise RuntimeError('No MO coefficients or Density found in FCHK')
        self._scf.mo_coeff, self._scf.mo_occ, self._scf.e_tot = self.mo_coeff, self.mo_occ, self.e_tot
        self._scf.is_fchk = True
        self._scf.path = path

    def make_rdm1(self, ao_repr=True):
        if self.mo_coeff is None: return None
        if isinstance(self.mo_coeff, list):
            ca, cb = self.mo_coeff; return np.array([np.dot(ca * self.mo_occ[0], ca.T), np.dot(cb * self.mo_occ[1], cb.T)])
        return np.dot(self.mo_coeff * self.mo_occ, self.mo_coeff.T)

    def get_ovlp(self): return self.mol.intor_symmetric('int1e_ovlp')

class FchkMolecule:
    def __init__(self, path):
        self.path = path
        with open(path, "r") as f: line1 = f.readline(); self.is_qchem = any(x in line1 for x in ["Q-Chem", "Q-CHEM", "Jobname.Temp"])
        if self.is_qchem: print(" | FCHK from Q-Chem")
        self.nalpha = int(read_from_fchk('Number of alpha electrons', path)[-1])
        self.mult = int(read_from_fchk('Multiplicity', path)[-1])
        beta_mo = read_from_fchk('Beta MO coefficients', path); self.unrestricted = (len(beta_mo) > 0)
        res_beta = read_from_fchk('Number of beta electrons', path); self.nbeta = int(res_beta[-1]) if res_beta else self.nalpha
        if self.mult != 1: self.unrestricted = True
        self.natoms = int(read_from_fchk('Number of atoms', path)[-1])
        self.ncshell = int(read_from_fchk('Number of contracted shells', path)[-1])
        self.charge = int(read_from_fchk('Charge', path)[-1])
        self.atomic_numbers = [int(i) for i in read_list_from_fchk('Atomic numbers', path)]
        res_nuc = read_list_from_fchk("Nuclear charges", path); self.nuclear_charges = [float(i) for i in res_nuc] if res_nuc else None
        self.atomic_symbols = read_atomic_symbols(self.atomic_numbers)
        self.coord = np.array(read_list_from_fchk('Current cartesian coordinates', path)).reshape(self.natoms, 3)
        try: self.e_tot = float(read_from_fchk('Total Energy', path)[-1])
        except: self.e_tot = 0.0
        self.mssh = [int(i) for i in read_list_from_fchk('Shell types', path)]
        self.cart = any(x >= 2 for x in self.mssh)
        self.mnsh = [int(i) for i in read_list_from_fchk('Number of primitives per shell', path)]
        self.iatsh = [int(i) for i in read_list_from_fchk('Shell to atom map', path)]
        self.expsh = read_list_from_fchk('Primitive exponents', path)
        self.c1 = read_list_from_fchk('Contraction coefficients', path)
        with open(path, 'r') as f: text = f.read()
        self.c2 = read_list_from_fchk('P(S=P) Contraction coefficients', path) if 'P(S=P)' in text else None
        res_mo, res_ao = read_from_fchk('Number of independent functions', path), read_from_fchk('Number of basis functions', path)
        self.numao = int(res_ao[-1]) if res_ao else 0; self.nummo = int(res_mo[-1]) if res_mo else self.numao

def make_basis(mf):
    ncshell, mssh, mnsh, iatsh, expsh, c1, c2 = mf.fchk.ncshell, mf.fchk.mssh, mf.fchk.mnsh, mf.fchk.iatsh, mf.fchk.expsh, mf.fchk.c1, mf.fchk.c2
    done_shells, exp_idx, coeff_idx, first_atom = {}, 0, 0, {}
    for atom_idx, sym in enumerate(mf.atomic_symbols):
        if sym not in first_atom: first_atom[sym] = atom_idx
    for i in range(ncshell):
        l_raw, atom_idx, n_prim = mssh[i], iatsh[i] - 1, mnsh[i]
        sym = mf.atomic_symbols[atom_idx]
        if atom_idx != first_atom[sym]:
            exp_idx += n_prim
            n_contr = (len(c1)-coeff_idx-sum(mnsh[k]*(2 if mssh[k]==-1 and not c2 else 1) for k in range(i+1, ncshell)))//n_prim if i<ncshell-1 else (len(c1)-coeff_idx)//n_prim
            coeff_idx += n_prim * n_contr; continue
        if sym not in done_shells: done_shells[sym] = []
        primitives = []
        if l_raw == -1:
            if c2:
                for _ in range(n_prim): primitives.append((expsh[exp_idx], c1[coeff_idx], c2[coeff_idx])); exp_idx += 1; coeff_idx += 1
            else:
                for j in range(n_prim): primitives.append((expsh[exp_idx+j], c1[coeff_idx+j], c1[coeff_idx+n_prim+j])); exp_idx += n_prim; coeff_idx += 2*n_prim
            done_shells[sym].append({"l": -1, "primitives": primitives})
        else:
            n_contr = (len(c1)-coeff_idx-sum(mnsh[k]*(2 if mssh[k]==-1 and not c2 else 1) for k in range(i+1, ncshell)))//n_prim if i<ncshell-1 else (len(c1)-coeff_idx)//n_prim
            cf = c1[coeff_idx: coeff_idx + n_prim * n_contr]
            for p in range(n_prim): primitives.append((expsh[exp_idx+p], [cf[j*n_prim+p] for j in range(n_contr)]))
            exp_idx += n_prim; coeff_idx += n_prim * n_contr; done_shells[sym].append({"l": abs(l_raw), "primitives": primitives})
    def order_shells(shell_list):
        ls, ml = {}, 0
        for d in shell_list:
            l, ps = d["l"], d["primitives"]; sh = [l]
            if l == -1:
                s, p = [0], [1]
                for e, cs, cp in ps: s.append([e, cs]); p.append([e, cp])
                ls.setdefault(0, []).append(s); ls.setdefault(1, []).append(p); ml = max(ml, 1)
            else:
                for e, c in ps: sh.append([e]+c if isinstance(c, list) else [e, c])
                ls.setdefault(l, []).append(sh); ml = max(ml, l)
        res = []
        for L in range(ml+1):
            if L in ls: res.extend(ls[L])
        return res
    return {sym: order_shells(sh) for sym, sh in done_shells.items()}

def readfchk(path, ecp=None):
    print(" | Reading FCHK file:", path)
    m = Mole2(path)
    m.ecp = ecp
    m.build()
    return m, MeanField2(path, m)._scf
