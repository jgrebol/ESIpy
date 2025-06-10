import unittest

from pyscf import gto, dft

import esipy

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
    basis='sto-3g',
    spin=0,
    charge=0,
)
mol.build()

mf = dft.RKS(mol)
mf.xc = 'B3LYP'
mf.kernel()

ring = [1, 2, 3, 4, 5, 6]


class ESItest(unittest.TestCase):

    def init_partition(self, part, exp):
        esitest = esipy.ESI(mol=mol, mf=mf, rings=ring, partition=part)
        self.assertEqual(esitest.partition, exp)

    def test_init_partition_mulliken(self):
        mulliken_parts = ["m", "M", "mul", "MUl", "MulLiKeN"]
        for part in mulliken_parts:
            with self.subTest(partition=part):
                self.init_partition(part, 'mulliken')

    def test_init_partition_lowdin(self):
        lowdin_parts = ["l", "L", "low", "LoW", "LoWdiN"]
        for part in lowdin_parts:
            with self.subTest(partition=part):
                self.init_partition(part, 'lowdin')

    def test_init_partition_meta_lowdin(self):
        meta_lowdin_parts = ["ml", "ML", "meta-low", "m-low", "mlow", "meta-lowdin", "metalowdin", "mlowdin",
                             "m-lowdin"]
        for part in meta_lowdin_parts:
            with self.subTest(partition=part):
                self.init_partition(part, 'meta_lowdin')

    def test_init_partition_nao(self):
        nao_parts = ["n", "nao", "natural"]
        for part in nao_parts:
            with self.subTest(partition=part):
                self.init_partition(part, 'nao')

    def test_init_partition_iao(self):
        iao_parts = ["i", "iao", "intrinsic"]
        for part in iao_parts:
            with self.subTest(partition=part):
                self.init_partition(part, 'iao')

    def test_init_mci_av1245(self):
        # For a 6MR: mci=True, av1245=False
        esitest = esipy.ESI(mol=mol, mf=mf, rings=[1, 2, 3, 4, 5, 6], partition='mulliken')
        self.assertTrue(esitest.mci)
        self.assertFalse(esitest.av1245)

        # For a 10MR: mci=True, av1245=True
        esitest = esipy.ESI(mol=mol, mf=mf, rings=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], partition='mulliken')
        self.assertTrue(esitest.mci)
        self.assertTrue(esitest.av1245)

        # For a 12MR: mci=False, av1245=True
        esitest = esipy.ESI(mol=mol, mf=mf, rings=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], partition='mulliken')
        self.assertFalse(esitest.mci)
        self.assertTrue(esitest.av1245)

        # If user specifies, it should be the option. We try the oppposite of what it would automatically be
        esitest = esipy.ESI(mci=False, av1245=True, mol=mol, mf=mf, rings=[1, 2, 3, 4, 5, 6], partition='mulliken')
        self.assertFalse(esitest.mci)
        self.assertTrue(esitest.av1245)

    def test_init_aoms(self):
        esitest = esipy.ESI(mol=mol, mf=mf, rings=[1, 2, 3, 4, 5, 6], partition='mulliken',
                            save='example01_mulliken')
        self.assertTrue(hasattr(esitest, "aom"))

        # We try getting the AOMs from a file
        # path = "/home/joan/DOCENCIA/Z-ESIpy/ESIpy-CLASS/esipy/tests/"
        # aom = path + "example01_mulliken.aoms"
        # esitest = esipy.ESI(aom=aom, rings=[1, 2, 3, 4, 5, 6], partition='mulliken')
        # self.assertTrue(hasattr(esitest, "aom"))
        # self.assertTrue(isinstance(esitest.aom, list))

    def test_init_molinfo(self):
        esitest = esipy.ESI(mol=mol, mf=mf, rings=[1, 2, 3, 4, 5, 6], partition='mulliken',
                            save='example01_mulliken')
        self.assertTrue(hasattr(esitest, "molinfo"))
        self.assertTrue(isinstance(esitest.molinfo, dict))

        # We try building it from 'mol' and 'mf'
        esitest.mol = mol
        esitest.mf = mf
        esitest.molinfo
        self.assertTrue(hasattr(esitest, "molinfo"))

        # We try getting the molinfo from a file
        # path = "/home/joan/DOCENCIA/Z-ESIpy/ESIpy-CLASS/esipy/tests/"
        # aom = path + "example01_mulliken.aoms"
        # molinfo = path + "example01_mulliken.molinfo"
        # esitest = esipy.ESI(aom=aom, molinfo=molinfo, rings=[1, 2, 3, 4, 5, 6], partition='mulliken')
        # self.assertTrue(hasattr(esitest, "molinfo"))
        # self.assertTrue(isinstance(esitest.molinfo, dict))


if __name__ == "__main__":
    unittest.main()
