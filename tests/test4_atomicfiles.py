import unittest
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import shutil
import numpy as np
from pyscf import gto, dft, mp, mcscf

import esipy

# Use a small molecule (Water) for fast unit testing of AOM reading/writing
mol = gto.M(
    atom='''
    O 0.000000000  0.000000000  0.119262000
    H 0.000000000  0.763239000 -0.477047000
    H 0.000000000 -0.763239000 -0.477047000
    ''',
    basis='sto-3g',
    spin=0,
    charge=0,
)
mol.build()

rest = dft.RKS(mol)
rest.xc = 'B3LYP'
rest.kernel()

rest_mp2 = mp.MP2(rest)
rest_mp2.kernel()

rest_cas = mcscf.CASSCF(rest, 4, 4)
rest_cas.kernel()

mol_unrest = mol.copy()
mol_unrest.spin = 2
mol_unrest.charge = 0
unrest = dft.UKS(mol_unrest)
unrest.xc = 'B3LYP'
unrest.kernel()

partitions_to_test = ['mulliken', 'lowdin', 'meta-lowdin', 'nao', 'iao']


def read_like_esi3d(path, mol, mul):
    """
    Reads a directory written by ESI.writeaoms() the way ESI-3D reads AIMAll files (esi_inp.f90):
    the .wfx gives the number of MOs, occupations and spin types; each <atom>.int needs a ' AIMInt' line,
    ' Integration is over atom <Symbol><n>', the AOM two records after ' The Atomic Overlap Matrix:'
    (lower triangle, or full rows for Mulliken) and ' Total electrons (N)' with the value in columns 50-66.
    Returns the AOMs, the occupations and the number of alpha MOs.
    """
    with open([os.path.join(path, f) for f in os.listdir(path) if f.endswith('.wfx')][0]) as f:
        wfx = f.read()
    block = lambda tag: wfx.split(f'<{tag}>\n')[1].split(f'</{tag}>')[0].splitlines()
    nmo = int(block('Number of Occupied Molecular Orbitals')[0])
    occ = np.array([float(x) for x in block('Molecular Orbital Occupation Numbers')])
    spins = block('Molecular Orbital Spin Types')
    assert len(occ) == len(spins) == nmo, "wfx: number of MOs, occupations and spin types differ"
    aoms = []
    for i in range(mol.natm):
        sym = mol.atom_symbol(i)
        with open(os.path.join(path, f"{sym.lower()}{i + 1}.int")) as f:
            lines = f.read().splitlines()
        assert not lines[0].startswith(' AIMInt') and any(l.startswith(' AIMInt') for l in lines)
        assert f" Integration is over atom {sym}{i + 1}" in lines
        start = lines.index(' The Atomic Overlap Matrix:') + 3
        rows = []
        for line in lines[start:]:
            if not line.strip():
                if rows:
                    break
                continue
            rows.append([float(x) for x in line.split()])
        s = np.zeros((nmo, nmo))
        for j, row in enumerate(rows):
            assert len(row) == (nmo if mul else j + 1)
            s[j, :len(row)] = row
        if not mul:
            s = s + np.tril(s, -1).T
        ntot = float(next(l for l in lines if l.startswith(' Total electrons (N)'))[49:66])
        np.testing.assert_allclose(ntot, np.dot(occ, np.diag(s)), atol=1e-8)
        aoms.append(s)
    return aoms, occ, sum(sp == ' Alpha' for sp in spins)


def flatten(x):
    """All the numbers of a (nested) AOM structure: restricted, [alpha, beta] or [AOMs, occupations]."""
    if isinstance(x, (list, tuple)):
        return np.concatenate([flatten(y) for y in x])
    return np.ravel(np.asarray(x, dtype=float))


def assert_esi3d_readable(path, mol, aom, mul):
    """The AOMs that ESI-3D would read must be the ones ESIpy wrote (restricted, unrestricted or NOs)."""
    from esipy.tools import wf_type
    wf = wf_type(aom)
    read, occ, nalpha = read_like_esi3d(path, mol, mul)
    for i, s in enumerate(read):
        if wf == "unrest":
            na = len(aom[0][i])
            assert nalpha == na
            np.testing.assert_allclose(s[:na, :na], aom[0][i], atol=1e-10)
            np.testing.assert_allclose(s[na:, na:], aom[1][i], atol=1e-10)
            np.testing.assert_allclose(s[:na, na:], 0.0, atol=1e-12)
        else:
            np.testing.assert_allclose(s, aom[0][i] if wf == "no" else aom[i], atol=1e-10)
    if wf == "no":
        np.testing.assert_allclose(occ, aom[1], atol=1e-10)
    else:
        np.testing.assert_allclose(occ, 2.0 if wf == "rest" else 1.0)

class ESItest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Change to the directory where the script is located
        cls.script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(cls.script_dir)
        cls.esipy_dir = os.path.join(cls.script_dir, "ATOMICFILES", "ESIpy")
        cls.aimall_dir = os.path.join(cls.script_dir, "ATOMICFILES", "AIMALL")
        os.makedirs(cls.esipy_dir, exist_ok=True)
        os.makedirs(cls.aimall_dir, exist_ok=True)

    def _test_method(self, mf, name_prefix):
        os.chdir(self.esipy_dir)
        from esipy.tools import format_short_partition
        for part in partitions_to_test:
            save_name = f"{name_prefix}_{part}"
            shortpart = format_short_partition(part)
            actual_readpath = f"{save_name}_{shortpart}_atomicfiles"
            
            # Write (from scratch: writeaoms keeps an existing .wfx)
            shutil.rmtree(actual_readpath, ignore_errors=True)
            esi_write = esipy.ESI(mol=mf.mol, mf=mf, partition=part, save=save_name)
            esi_write.writeaoms(save_name)
            assert_esi3d_readable(actual_readpath, mf.mol, esi_write.aom, part == 'mulliken')

            # Read
            esi_read = esipy.ESI(read=True, molinfo=f"{save_name}.molinfo", partition=part, name=name_prefix, readpath=actual_readpath)
            esi_read.readaoms()
            
            # Extract matrices safely
            if isinstance(esi_write.aom, tuple):
                write_aom_data = esi_write.aom[0]
            elif isinstance(esi_write.aom, list) and len(esi_write.aom) == 2 and isinstance(esi_write.aom[1], np.ndarray):
                write_aom_data = esi_write.aom[0]
            else:
                write_aom_data = esi_write.aom
                
            if isinstance(esi_read.aom, tuple) or (isinstance(esi_read.aom, list) and len(esi_read.aom) == 2 and isinstance(esi_read.aom[1], np.ndarray)):
                read_aom_data = esi_read.aom[0]
            else:
                read_aom_data = esi_read.aom
                
            # Recursive check for unrest lists [alpha, beta]
            def assert_aoms_match(w, r):
                if isinstance(w, list) and len(w) > 0 and isinstance(w[0], list):
                    for w_spin, r_spin in zip(w, r):
                        assert_aoms_match(w_spin, r_spin)
                else:
                    for mat_w, mat_r in zip(w, r):
                        np.testing.assert_allclose(mat_w, mat_r, atol=1e-6)
            
            assert_aoms_match(write_aom_data, read_aom_data)
        os.chdir(self.script_dir)

    def test_rks(self):
        self._test_method(rest, "test_rks")

    def test_uks(self):
        self._test_method(unrest, "test_uks")
        
    def test_mp2(self):
        self._test_method(rest_mp2, "test_mp2")
        
    def test_cas(self):
        self._test_method(rest_cas, "test_cas")

    def test_aimall_read(self):
        # Read user-provided AIMALL tests
        aimall_subdirs = [d for d in os.listdir(self.aimall_dir) if os.path.isdir(os.path.join(self.aimall_dir, d))]
        run_count = 0
        for d in aimall_subdirs:
            adir = os.path.join(self.aimall_dir, d)
            int_files = [f for f in os.listdir(adir) if f.endswith('.int')]
            if not int_files:
                continue
            run_count += 1
            prefix = d.replace("_atomicfiles", "")
            fchk_path = os.path.join(self.aimall_dir, prefix + ".fchk")
            
            try:
                if os.path.exists(fchk_path):
                    esi_read = esipy.ESI(fchk=fchk_path, read=True, readpath=adir, partition="qtaim")
                else:
                    esi_read = esipy.ESI(read=True, readpath=adir, partition="qtaim")
                    
                esi_read.readaoms()
                self.assertTrue(len(esi_read.aom) > 0)

                # Rewrite the AIMAll AOMs with ESIpy: they must read back identically and be readable by ESI-3D
                if os.path.exists(fchk_path):
                    from esipy.atomicfiles import write_aoms, read_aoms
                    from esipy.readfchk import readfchk
                    fmol, fmf = readfchk(fchk_path)
                    os.chdir(self.esipy_dir)
                    shutil.rmtree(f"test4_aimall_{prefix}_qtaim_atomicfiles", ignore_errors=True)
                    write_aoms(fmol, fmf, f"test4_aimall_{prefix}", esi_read.aom, partition="qtaim")
                    rewritten = read_aoms(f"test4_aimall_{prefix}_qtaim_atomicfiles")
                    os.chdir(self.script_dir)
                    np.testing.assert_allclose(flatten(rewritten), flatten(esi_read.aom), atol=1e-12)
                    assert_esi3d_readable(os.path.join(self.esipy_dir, f"test4_aimall_{prefix}_qtaim_atomicfiles"),
                                          fmol, esi_read.aom, False)
            except Exception as e:
                self.fail(f"Failed reading AIMALL directory {adir} (FCHK: {os.path.exists(fchk_path)}): {e}")
        if run_count == 0:
            self.skipTest("No AIMAll .int files found in the test directory.")


if __name__ == "__main__":
    unittest.main()
