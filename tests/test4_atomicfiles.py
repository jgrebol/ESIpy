import unittest
import os
import shutil
import numpy as np
from pyscf import gto, dft
import esipy

class TestAtomicFiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(atom='H 0 0 0; H 0 0 1.5', basis='sto-3g', verbose=0)
        cls.mf = dft.RKS(cls.mol); cls.mf.xc = 'lda'; cls.mf.kernel()
        cls.test_dir = os.path.dirname(os.path.abspath(__file__))

    def setUp(self):
        os.chdir(self.test_dir)

    def test_basic_write_read(self):
        name = 'test_h2'
        esi = esipy.ESI(mol=self.mol, mf=self.mf, partition='iao')
        out_dir = name + '_iao_atomicfiles'
        if os.path.exists(out_dir): shutil.rmtree(out_dir)
        esi.writeaoms(name)
        self.assertTrue(os.path.exists(out_dir))
        shutil.rmtree(out_dir)

    def test_fragments_and_rings(self):
        # 4 carbon square (24 electrons)
        mol4 = gto.M(atom='C 0 0 0; C 1.4 0 0; C 1.4 1.4 0; C 0 1.4 0', basis='sto-3g', verbose=0)
        mf4 = dft.RKS(mol4); mf4.kernel()
        
        # Ring with a fragment: atoms 1 and 2 as a set, plus atoms 3 and 4
        # This forms a 3-center ring: Fragment{1,2} - Atom 3 - Atom 4
        ring_with_frag = [{1, 2}, 3, 4]
        
        esi = esipy.ESI(mol=mol4, mf=mf4, rings=[ring_with_frag], partition='nao')
        
        self.assertEqual(len(esi.indicators), 1)
        self.assertGreater(esi.indicators[0].mci, 0)
        
        name = 'test_frag'
        out_dir = name + '_nao_atomicfiles'
        if os.path.exists(out_dir): shutil.rmtree(out_dir)
        esi.writeaoms(name)
        self.assertTrue(os.path.exists(out_dir))
        shutil.rmtree(out_dir)

if __name__ == '__main__':
    unittest.main()
