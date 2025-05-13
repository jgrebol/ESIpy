import unittest
from esipy import ESI
from pyscf import gto, scf

# Molecule and PySCF setup from the original test7_rings.py
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
mf.max_cycle = 1
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
        self.assertEqual(self.esi.rings, [[1, 2, 3, 4, 5, 6]], "Failed for integer rings")

    def test_morering(self):
        """Test rings as integers."""
        rings_data = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
        self.esi = ESI(mol=mol, mf=mf, partition="nao", rings=rings_data)
        self.esi.print()
        self.assertEqual(self.esi.rings, [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], "Failed for integer rings")

    def test_onefrag(self):
        """Test rings as fragments (set of atom labels)."""
        rings_data = [{1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10}, {11, 12}]
        self.esi = ESI(mol=mol, mf=mf, partition="nao", rings=rings_data)
        self.esi.print()
        print(self.esi.rings)
        self.assertEqual(self.esi.rings, [[{1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10}, {11, 12}]], "Failed for fragment rings")

    def test_twofrags(self):
        """Test rings as fragments (set of atom labels)."""
        rings_data = [[{1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10}, {11, 12}], [{13, 14}, {15, 16}, {17, 18}, {19, 20}, {21, 22}, {23, 24}]]
        self.esi = ESI(mol=mol, mf=mf, partition="nao", rings=rings_data)
        print(self.esi.rings)
        self.assertEqual(self.esi.rings, [[{1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10}, {11, 12}], [{13, 14}, {15, 16}, {17, 18}, {19, 20}, {21, 22}, {23, 24}]], "Failed for fragment rings")

    def test_atomandfrag(self):
        """Test rings as fragments (set of atom labels)."""
        rings_data = [{1, 2}, 3, {4, 5}, 6, {7, 8}, 9]
        self.esi = ESI(mol=mol, mf=mf, partition="nao", rings=rings_data)
        self.assertEqual(self.esi.rings, [[{1, 2}, 3, {4, 5}, 6, {7, 8}, 9]], "Failed for fragment rings")

    def test_rings_mixed(self):
        """Test rings as a mix of integers and fragments."""
        rings_data = [3, [1, 2, 3]]
        self.esi.rings = rings_data
        self.assertEqual(self.esi.rings, [[3, [1, 2, 3]]], "Failed for mixed rings")

    def test_rings_nested(self):
        """Test rings as a list of lists (nested rings)."""
        rings_data = [[3, 4], [[1, 2, 3], [4, 5, 6]]]
        self.esi.rings = rings_data
        self.assertEqual(self.esi.rings, rings_data, "Failed for nested rings")

    def test_findrings(self):
        """Test findrings method."""
        rings_data = "f"
        self.esi = ESI(mol=mol, mf=mf, partition="nao", rings=rings_data)


if __name__ == "__main__":
    unittest.main()
