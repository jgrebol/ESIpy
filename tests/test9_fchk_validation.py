import unittest
import os
import numpy as np
from esipy.readfchk import readfchk
from esipy import ESI
from pyscf import gto, scf

class TestFchkValidation(unittest.TestCase):
    def setUp(self):
        # Path relative to the tests directory where this file resides
        self.fchk_dir = os.path.join(os.path.dirname(__file__), 'FCHK')
        self.systems = {
            'H2O': {
                'path': os.path.join(self.fchk_dir, 'GAUSSIAN', 'h2o_sto3g.fchk'),
                'rings': None,
                'nelec': 10
            },
            'Benzene': {
                'path': os.path.join(self.fchk_dir, 'GAUSSIAN', 'bz.fchk'),
                'rings': [[1, 2, 3, 4, 5, 6]],
                'nelec': 42
            }
        }

    def run_full_comparison(self, system_name, partition):
        sys_info = self.systems[system_name]
        path = sys_info['path']
        
        if not os.path.exists(path):
            self.skipTest(f'FCHK file {path} not found')

        mol, mf = readfchk(path)
        esi = ESI(mol=mol, mf=mf, partition=partition, rings=sys_info['rings'])
        
        trace_sum = np.sum([np.trace(m) for m in esi.aom])
        self.assertAlmostEqual(trace_sum, sys_info['nelec']/2.0, delta=0.01)
        
        if esi.rings:
            mci = esi.indicators[0].mci
            self.assertGreater(mci, 0)

    def test_h2o_iao(self): self.run_full_comparison('H2O', 'iao')
    def test_benzene_iao(self): self.run_full_comparison('Benzene', 'iao')
    def test_benzene_nao(self): self.run_full_comparison('Benzene', 'nao')

if __name__ == '__main__':
    unittest.main()
