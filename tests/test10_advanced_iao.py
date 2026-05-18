import unittest
import os
import numpy as np
from esipy.readfchk import readfchk
from esipy.make_aoms import make_aoms
from esipy import ESI

class TestAdvancedIAO(unittest.TestCase):
    def setUp(self):
        self.fchk_dir = os.path.join(os.path.dirname(__file__), '..', 'FCHK')
        self.h2o_fchk = os.path.join(self.fchk_dir, 'GAUSSIAN', 'h2o.fchk')
        self.bz_fchk = os.path.join(self.fchk_dir, 'GAUSSIAN', 'bz.fchk')

    def test_effao_construction(self):
        if not os.path.exists(self.h2o_fchk):
            self.skipTest('h2o.fchk not found')
        
        mol, mf = readfchk(self.h2o_fchk)
        aoms = make_aoms(mol, mf, 'iao-effao-nao')
        self.assertEqual(len(aoms), mol.natm)
        total_trace = sum(np.trace(m) for m in aoms)
        self.assertAlmostEqual(total_trace, 5.0, delta=1e-3)

    def test_peiao_construction(self):
        if not os.path.exists(self.h2o_fchk):
            self.skipTest('h2o.fchk not found')
        
        mol, mf = readfchk(self.h2o_fchk)
        aoms = make_aoms(mol, mf, 'peiao')
        self.assertEqual(len(aoms), mol.natm)
        total_trace = sum(np.trace(m) for m in aoms)
        self.assertAlmostEqual(total_trace, 5.0, delta=1e-3)

    def test_dpeiao_blending(self):
        if not os.path.exists(self.h2o_fchk):
            self.skipTest('h2o.fchk not found')
        
        mol, mf = readfchk(self.h2o_fchk)
        aoms = make_aoms(mol, mf, 'dpeiao(0.5)')
        self.assertEqual(len(aoms), mol.natm)
        total_trace = sum(np.trace(m) for m in aoms)
        self.assertAlmostEqual(total_trace, 5.0, delta=1e-3)

    def test_wiao_construction(self):
        if not os.path.exists(self.h2o_fchk):
            self.skipTest('h2o.fchk not found')
        
        mol, mf = readfchk(self.h2o_fchk)
        aoms = make_aoms(mol, mf, 'wiao')
        self.assertEqual(len(aoms), mol.natm)
        total_trace = sum(np.trace(m) for m in aoms)
        self.assertAlmostEqual(total_trace, 5.0, delta=1e-3)

    def test_multiprocessing_mci(self):
        if not os.path.exists(self.bz_fchk):
            self.skipTest('bz.fchk not found')
            
        mol_bz, mf_bz = readfchk(self.bz_fchk)
        ring = [1, 2, 3, 4, 5, 6]
        
        esi_seq = ESI(mol=mol_bz, mf=mf_bz, partition='iao', rings=ring, ncores=1)
        mci_seq = esi_seq.indicators[0].mci
        
        esi_par = ESI(mol=mol_bz, mf=mf_bz, partition='iao', rings=ring, ncores=2)
        mci_par = esi_par.indicators[0].mci
        
        self.assertAlmostEqual(mci_seq, mci_par, places=6)

    def test_natorb_fchk_format(self):
        # Use the CASSCF FCHK if available on the system (for local testing)
        fchk_path = '/home/joan/PycharmProjects/ESIpy/joan/LiH/GS/lih_1.5.fchk'
        if not os.path.exists(fchk_path):
            self.skipTest('CASSCF FCHK not found at ' + fchk_path)
            
        mol, mf = readfchk(fchk_path)
        aoms = make_aoms(mol, mf, 'iao')
        
        # Check for Natural Orbitals format [aoms, occ_matrix]
        self.assertIsInstance(aoms, list)
        self.assertEqual(len(aoms), 2)
        self.assertIsInstance(aoms[1], np.ndarray)

if __name__ == '__main__':
    unittest.main()
