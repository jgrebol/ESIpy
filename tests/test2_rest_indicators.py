import unittest
from esipy import ESI
from esipy.indicators import *
from numpy import trace, dot
from pyscf import gto, dft

# Set up molecule and DFT calculations
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
    ''', basis='sto-3g', spin=0, charge=0)
mol.build()
mf = dft.RKS(mol)
mf.xc = 'B3LYP'
mf.kernel()
ring = [1, 2, 3, 4, 5, 6]

# Expected results for different partitions
expected = {
    'mulliken': dict(exp_pop_atm1=6.0805, exp_li_atm1=4.0933, exp_di_1_all=1.9871,
                     exp_lis_sum=27.0966, exp_dis_sum=14.9034, exp_di_12=1.4338,
                     exp_iring=0.088444, exp_mci=0.132157,
                     exp_av=16.613, exp_pdi=0.116630),
    'lowdin': dict(exp_pop_atm1=6.0437, exp_li_atm1=4.0448, exp_di_1_all=1.9989,
                     exp_lis_sum=27.0122, exp_dis_sum=14.9878, exp_di_12=1.4439,
                     exp_iring=0.088425, exp_mci=0.132116,
                     exp_av=16.614, exp_pdi=0.116795),
    'meta_lowdin': dict(exp_pop_atm1=6.0488, exp_li_atm1=4.0503, exp_di_1_all=1.9985,
                   exp_lis_sum=27.0162, exp_dis_sum=14.9838, exp_di_12=1.4439,
                   exp_iring=0.088434, exp_mci=0.132119,
                   exp_av=16.616, exp_pdi=0.116896),
    'nao': dict(exp_pop_atm1=6.0548, exp_li_atm1=4.0604, exp_di_1_all=1.9944,
                        exp_lis_sum=27.0428, exp_dis_sum=14.9572, exp_di_12=1.4397,
                        exp_iring=0.088477, exp_mci=0.132148,
                        exp_av=16.621, exp_pdi=0.117195),
    'iao': dict(exp_pop_atm1=6.1355, exp_li_atm1=4.1506, exp_di_1_all=1.9848,
                exp_lis_sum=27.1461, exp_dis_sum=14.8539, exp_di_12=1.4364,
                exp_iring=0.088283, exp_mci=0.132127,
                exp_av=16.560, exp_pdi=0.113278)
}

class ESItest(unittest.TestCase):

    def run_pop_tests(self, partition, exp):
        esitest = ESI(mol=mol, mf=mf, rings=ring, partition=partition)
        aom = esitest.Smo

        exp_pop_atm1 = exp['exp_pop_atm1']
        exp_li_atm1 = exp['exp_li_atm1']
        exp_di_1_all = exp['exp_di_1_all']
        exp_lis_sum = exp['exp_lis_sum']
        exp_dis_sum = exp['exp_dis_sum']
        exp_di_12 = exp['exp_di_12']

        # Testing atomic populations of C1
        self.assertAlmostEqual(2 * trace(aom[0]), exp_pop_atm1, places=4)
        # Testing LI for C1
        self.assertAlmostEqual(2 * trace(dot(aom[0], aom[0])), exp_li_atm1, places=4)
        # Testing delocalized electrons for C1
        di_1_all = sum(trace(dot(aom[0], aom[i])) for i in range(1, 12))
        self.assertAlmostEqual(2 * di_1_all, exp_di_1_all, places=4)
        # Testing sum of DI and LI
        lis = [2 * trace(dot(aom[i], aom[i])) for i in range(len(aom))]
        dis = [2 * trace(dot(aom[i], aom[j])) for i in range(len(aom)) for j in range(len(aom)) if i != j]
        self.assertAlmostEqual(sum(lis), exp_lis_sum, places=4)
        self.assertAlmostEqual(sum(dis), exp_dis_sum, places=4)
        self.assertAlmostEqual(sum(lis) + sum(dis), 42, places=4)
        # Testing DI between C1 and C2
        self.assertAlmostEqual(4 * trace(dot(aom[0], aom[1])), exp_di_12, places=4)

    def run_indicator_tests(self, partition, exp):
        esitest = ESI(mol=mol, mf=mf, rings=ring, partition=partition)
        aom = esitest.Smo

        exp_iring = exp['exp_iring']
        exp_mci = exp['exp_mci']
        exp_av = exp['exp_av']
        exp_pdi = exp['exp_pdi']

        # Multicenter indicators
        self.assertAlmostEqual(2 * compute_iring(ring, aom), exp_iring, places=6)
        self.assertAlmostEqual(2 * sequential_mci(ring, aom, partition=partition), exp_mci, places=5)
        self.assertAlmostEqual(2 * multiprocessing_mci(ring, aom, ncores=1, partition=partition), exp_mci, places=5)
        self.assertAlmostEqual(2 * compute_av1245(ring, aom)[0], exp_av, places=2)
        esitest.av1245 = True
        esitest.calc()
        self.assertAlmostEqual(2 * compute_pdi(ring, aom)[0], exp_pdi, places=6)

        #Geometric indicators
        homa, en, geo = compute_homa(ring, mol)
        self.assertAlmostEqual(homa, 0.993307, places=6)
        self.assertAlmostEqual(en, 0.006693, places=6)
        self.assertAlmostEqual(geo, 0, places=6)
        self.assertAlmostEqual(compute_bla(ring, mol)[0], 0, places=6)

    def test_mulliken(self):
        partition = 'mulliken'
        exp = expected[partition]
        self.run_pop_tests(partition, exp=exp)
        self.run_indicator_tests(partition, exp=exp)


    def test_lowdin(self):
        partition = 'lowdin'
        exp = expected[partition]
        self.run_pop_tests(partition, exp=exp)
        self.run_indicator_tests(partition, exp=exp)

    def test_meta_lowdin(self):
        partition = 'meta_lowdin'
        exp = expected[partition]
        self.run_pop_tests(partition, exp=exp)
        self.run_indicator_tests(partition, exp=exp)

    def test_nao(self):
        partition = 'nao'
        exp = expected[partition]
        self.run_pop_tests(partition, exp=exp)
        self.run_indicator_tests(partition, exp=exp)

    def test_iao(self):
        partition = 'iao'
        exp = expected[partition]
        self.run_pop_tests(partition, exp=exp)
        self.run_indicator_tests(partition, exp=exp)


if __name__ == "__main__":
    unittest.main()

