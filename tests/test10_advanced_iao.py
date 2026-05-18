import unittest
import os
import numpy as np
from esipy.readfchk import readfchk
from esipy.make_aoms import make_aoms
from esipy import ESI

class TestAdvancedIAO(unittest.TestCase):
    def setUp(self):
        self.fchk_dir = os.path.join(os.path.dirname(__file__), 'FCHK')
        self.bz_fchk = os.path.join(self.fchk_dir, 'GAUSSIAN', 'bz.fchk')
        self.cas_fchk = os.path.join(self.fchk_dir, 'GAUSSIAN', 'lih_cas.fchk')

    def test_effao_construction(self):
        if not os.path.exists(self.bz_fchk):
            self.skipTest('bz.fchk not found')
        
        mol, mf = readfchk(self.bz_fchk)
        aoms = make_aoms(mol, mf, 'iao-effao-nao')
        self.assertEqual(len(aoms), mol.natm)
        total_trace = sum(np.trace(m) for m in aoms)
        self.assertAlmostEqual(total_trace, 21.0, delta=0.1)

    def test_peiao_construction(self):
        if not os.path.exists(self.bz_fchk):
            self.skipTest('bz.fchk not found')
        
        mol, mf = readfchk(self.bz_fchk)
        aoms = make_aoms(mol, mf, 'peiao')
        self.assertEqual(len(aoms), mol.natm)
        total_trace = sum(np.trace(m) for m in aoms)
        self.assertAlmostEqual(total_trace, 21.0, delta=0.1)

    def test_dpeiao_blending(self):
        if not os.path.exists(self.bz_fchk):
            self.skipTest('bz.fchk not found')
        
        mol, mf = readfchk(self.bz_fchk)
        aoms = make_aoms(mol, mf, 'dpeiao(0.5)')
        self.assertEqual(len(aoms), mol.natm)
        total_trace = sum(np.trace(m) for m in aoms)
        self.assertAlmostEqual(total_trace, 21.0, delta=0.1)

    def test_natorb_fchk_format(self):
        if not os.path.exists(self.cas_fchk):
            self.skipTest('CASSCF FCHK not found')
            
        mol, mf = readfchk(self.cas_fchk)
        aoms = make_aoms(mol, mf, 'iao')
        self.assertIsInstance(aoms, list)
        self.assertEqual(len(aoms), 2)
        self.assertIsInstance(aoms[1], np.ndarray)

if __name__ == '__main__':
    unittest.main()
