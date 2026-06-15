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
            
            # Write
            esi_write = esipy.ESI(mol=mf.mol, mf=mf, partition=part, save=save_name)
            esi_write.writeaoms(save_name)
            
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
            except Exception as e:
                self.fail(f"Failed reading AIMALL directory {adir} (FCHK: {os.path.exists(fchk_path)}): {e}")
        if run_count == 0:
            self.skipTest("No AIMAll .int files found in the test directory.")


if __name__ == "__main__":
    unittest.main()
