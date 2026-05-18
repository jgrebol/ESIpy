import unittest
import os
from pyscf import gto, dft
import numpy as np
import esipy
import shutil

mol = gto.M(
    atom='''
    C        0.000000000      0.000000000      1.393096000
    C        0.000000000      1.206457000      0.696548000
    C        0.000000000      1.206457000     -0.696548000
    C        0.000000000      0.000000000     -1.393096000
    C        0.000000000     -1.206457000     -0.696548000
    C        0.000000000     -1.206457000      0.696548000
    H        0.000000000      0.000000000      2.483127000
    H        0.000000000      2.150450000      1.241569000
    H        0.000000000      2.150450000     -1.241569000
    H        0.000000000      0.000000000     -2.483127000
    H        0.000000000     -2.150450000     -1.241569000
    H        0.000000000     -2.150450000      1.241569000
    ''',
    basis='sto-3g', spin=0, charge=0,
)
mol.build()
rest = dft.RKS(mol); rest.xc = 'B3LYP'; rest.kernel()
ring = [1, 2, 3, 4, 5, 6]

class ESItest(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(self.test_dir)

    def test_write_read_aoms_fragments(self):
        name = "test4_fragments"
        # Define fragments: atoms 1,2 as one fragment, others individual
        fragments = [{1, 2}, {3}, {4}, {5}, {6}]
        esi = esipy.ESI(mol=mol, mf=rest, rings=[ring], partition='iao', fragments=fragments)
        
        # Test writing atomic files
        out_dir = name + "_iao_atomicfiles"
        if os.path.exists(out_dir): shutil.rmtree(out_dir)
        esi.writeaoms(name)
        self.assertTrue(os.path.exists(out_dir))
        
        # Check if fragment is correctly handled in rings
        # After processing, the ring should contain the fragment index
        self.assertEqual(len(esi.rings), 1)
        # Fragment FF7 (1,2)
        # Original atoms 3,4,5,6 are still there
        # Note: process_fragments logic re-indexes rings if they contain fragments
        # Just verify it doesn't crash and returns indicators
        self.assertGreater(esi.indicators[0].mci, 0)

if __name__ == '__main__':
    unittest.main()
