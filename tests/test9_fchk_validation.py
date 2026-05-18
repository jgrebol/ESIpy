import unittest
import os
import numpy as np
from esipy.readfchk import readfchk
from esipy import ESI
from pyscf import gto, scf

class TestFchkValidation(unittest.TestCase):
    def setUp(self):
        self.fchk_dir = os.path.join(os.path.dirname(__file__), 'FCHK')
        self.h2o_fchk = os.path.join(self.fchk_dir, 'GAUSSIAN', 'h2o_sto3g.fchk')

    def test_h2o_reading(self):
        if not os.path.exists(self.h2o_fchk):
            self.skipTest('h2o_sto3g.fchk not found')
        
        mol, mf = readfchk(self.h2o_fchk)
        self.assertEqual(mol.natm, 3)
        self.assertAlmostEqual(mf.e_tot, -74.96, delta=0.1)

    def test_h2o_iao_main(self):
        if not os.path.exists(self.h2o_fchk):
            self.skipTest('h2o_sto3g.fchk not found')
            
        mol, mf = readfchk(self.h2o_fchk)
        esi = ESI(mol=mol, mf=mf, partition='iao')
        trace_sum = sum(np.trace(m) for m in esi.aom)
        self.assertAlmostEqual(trace_sum, 5.0, delta=1e-3)

if __name__ == '__main__':
    unittest.main()
