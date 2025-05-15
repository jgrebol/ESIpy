import unittest
from esipy import ESI
from pyscf import gto, scf

mol = gto.Mole()
mol.atom = '''
6       -0.340927711      0.000000000     -3.650697859
6       -0.340927711      0.948773522     -2.588797782
6       -0.340927711      0.475397034     -1.248555259
6       -0.340927711     -0.927940074     -0.987279681
6       -0.340927711     -1.851936189     -2.067357638
6       -0.340927711     -1.348900139     -3.399557226
6       -0.340927711      1.403337108     -0.163867660
6       -0.340927711     -1.403337108      0.358683495
6       -0.340927711     -0.475397034      1.443371094
6       -0.340927711      0.927940074      1.182095516
6       -0.340927711     -0.948773522      2.783613616
6       -0.340927711     -2.354010817      3.014072106
6       -0.340927711     -3.245954951      1.971460546
6       -0.340927711     -2.800710044      0.618849004
6       -0.340927711     -3.702910013     -0.482892705
6       -0.340927711     -3.245954008     -1.776644898
1       -0.340927711     -3.950336652     -2.604267108
1       -0.340927711     -4.770830586     -0.281251730
1       -0.340927711      0.364550104     -4.674521419
1       -0.340927711     -2.057485527     -4.223582793
1       -0.340927711     -2.713344726      4.039738037
1       -0.340927711     -4.314887089      2.167663171
6       -0.340927711      2.354010817     -2.819256271
6       -0.340927711      3.245954951     -1.776644711
6       -0.340927711      2.800710044     -0.424033169
1       -0.340927711      2.713344726     -3.844922202
1       -0.340927711      4.314887089     -1.972847337
6       -0.340927711      3.702910013      0.677708540
6       -0.340927711      3.245954008      1.971460732
6       -0.340927711      1.851936189      2.262173473
1       -0.340927711      4.770830586      0.476067565
1       -0.340927711      3.950336652      2.799082943
6       -0.340927711      0.000000000      3.845513694
1       -0.340927711     -0.364550104      4.869337254
6       -0.340927711      1.348900139      3.594373061
1       -0.340927711      2.057485527      4.418398627
'''
mol.basis = 'sto-3g'
mol.spin = 0
mol.charge = 0
mol.symmetry = True
mol.verbose = 0
mol.max_memory = 4000
mol.build()

mf = scf.HF(mol)
mf.max_cycle = 1 # Only for ESIpy qualitative purpose
mf.kernel()

class TestRingsBehavior(unittest.TestCase):

    def setUp(self):
        """Set up for test methods."""
        self.esi = ESI(mol=mol, mf=mf, partition="nao")

    def test_onering(self):
        """Test rings as integers."""
        rings_data = [1, 2, 3, 4, 5, 6]
        self.esi = ESI(mol=mol, mf=mf, partition="nao", rings=rings_data)
        self.esi.print()
        self.assertEqual(self.esi.filtrings, [[1, 2, 3, 4, 5, 6]], "Failed for integer rings")

    def test_morering(self):
        """Test rings as integers."""
        rings_data = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
        self.esi = ESI(mol=mol, mf=mf, partition="nao", rings=rings_data)
        self.esi.print()
        self.assertEqual(self.esi.filtrings, [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], "Failed for integer rings")

    def test_onefrag(self):
        """Test rings as fragments (set of atom labels)."""
        rings_data = [{1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10}, {11, 12}]
        self.esi = ESI(mol=mol, mf=mf, partition="nao", rings=rings_data)
        self.esi.print()
        print(self.esi.rings)
        self.assertEqual(self.esi.filtrings, [[37, 38, 39, 40, 41, 42]], "Failed for fragment rings")

    def test_twofrags(self):
        """Test rings as fragments (set of atom labels)."""
        rings_data = [[{1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10}, {11, 12}],[{1, 12}, {2, 11}, {3, 10}, {4, 9}, {5, 8}, {6, 7}]]
        self.esi = ESI(mol=mol, mf=mf, partition="nao", rings=rings_data)
        print(self.esi.rings)
        self.assertEqual(self.esi.filtrings, [[37, 38, 39, 40, 41, 42], [43, 44, 45, 46, 47, 48]], "Failed for two fragments")

    def test_atomandfrag(self):
        """Test rings as fragments (set of atom labels)."""
        rings_data = [{1, 2}, 3, {4, 5}, 6, {7, 8}, 9]
        self.esi = ESI(mol=mol, mf=mf, partition="nao", rings=rings_data)
        self.assertEqual(self.esi.filtrings, [[37, 3, 38, 6, 39, 9]], "Failed for fragment rings")

    def test_findrings(self):
        """Test ring finding algorithm"""
        rings = [
                 [2, 3, 7, 25, 24, 23],
         [2, 3, 7, 10, 30, 29, 28, 25, 24, 23],
         [2, 3, 4, 8, 9, 10, 30, 29, 28, 25, 24, 23],
         [2, 3, 4, 8, 9, 10, 7, 25, 24, 23],
         [2, 1, 6, 5, 16, 15, 14, 8, 9, 10, 7, 3],
         [2, 1, 6, 5, 16, 15, 14, 8, 4, 3],
         [2, 1, 6, 5, 4, 8, 9, 10, 7, 25, 24, 23],
         [2, 1, 6, 5, 4, 8, 9, 10, 7, 3],
         [2, 1, 6, 5, 4, 3],
         [2, 1, 6, 5, 4, 3, 7, 25, 24, 23],
         [3, 4, 8, 14, 13, 12, 11, 33, 35, 30, 10, 7],
         [3, 4, 8, 14, 13, 12, 11, 9, 10, 7],
         [3, 4, 8, 9, 11, 33, 35, 30, 29, 28, 25, 7],
         [3, 4, 8, 9, 11, 33, 35, 30, 10, 7],
         [3, 4, 8, 9, 10, 30, 29, 28, 25, 7],
         [3, 4, 8, 9, 10, 7],
         [3, 4, 5, 16, 15, 14, 13, 12, 11, 9, 10, 7],
         [3, 4, 5, 16, 15, 14, 8, 9, 10, 7],
         [4, 5, 16, 15, 14, 13, 12, 11, 9, 8],
         [4, 5, 16, 15, 14, 8],
         [7, 10, 30, 29, 28, 25],
         [7, 10, 9, 11, 33, 35, 30, 29, 28, 25],
         [8, 9, 11, 12, 13, 14],
         [8, 9, 10, 30, 35, 33, 11, 12, 13, 14],
         [9, 10, 30, 35, 33, 11],
        ]
        self.esi = ESI(mol=mol, mf=mf, partition="nao", rings="f")
        self.assertEqual(self.esi.filtrings, rings, "Failed for ring finding algorithm")

if __name__ == "__main__":
    unittest.main()
