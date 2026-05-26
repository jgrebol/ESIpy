import unittest
import numpy as np
from numpy import trace
from numpy.linalg import multi_dot
from pyscf import gto, scf, mp
from esipy import ESI

# Set up molecule and MP2 calculation
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

myhf = scf.RHF(mol)
myhf.kernel()

mf = mp.MP2(myhf)
mf.kernel()

ring = [1, 2, 3, 4, 5, 6]

expected = {
    "mulliken": {
        "exp_pop_atm1": 6.0613,
        "exp_lif_atm1": 4.4857,
        "exp_lix_atm1": 4.0725,
        "exp_lifs_sum": 30.0244,
        "exp_lixs_sum": 27.0785,
        "exp_difs_sum": 11.9756,
        "exp_dixs_sum": 14.2745,
        "exp_dif_12": 1.1239,
        "exp_dix_12": 1.3615,
        "exp_iring": 0.264557,
        "exp_mci": 0.385638,
        "exp_av": 49.3906,
        "exp_pdi": 0.258353,
        "Nx": 41.3529
    },
    "lowdin": {
        "exp_pop_atm1": 6.03,
        "exp_lif_atm1": 4.4452,
        "exp_lix_atm1": 4.0305,
        "exp_lifs_sum": 29.9608,
        "exp_lixs_sum": 27.0054,
        "exp_difs_sum": 12.0392,
        "exp_dixs_sum": 14.3475,
        "exp_dif_12": 1.1321,
        "exp_dix_12": 1.3711,
        "exp_iring": 0.264629,
        "exp_mci": 0.38564,
        "exp_av": 49.4074,
        "exp_pdi": 0.25912,
        "Nx": 41.3529
    },
    "meta-lowdin": {
        "exp_pop_atm1": 6.0351,
        "exp_lif_atm1": 4.4507,
        "exp_lix_atm1": 4.036,
        "exp_lifs_sum": 29.9639,
        "exp_lixs_sum": 27.0089,
        "exp_difs_sum": 12.0361,
        "exp_dixs_sum": 14.344,
        "exp_dif_12": 1.1321,
        "exp_dix_12": 1.3711,
        "exp_iring": 0.264659,
        "exp_mci": 0.38565,
        "exp_av": 49.4114,
        "exp_pdi": 0.259402,
        "Nx": 41.3529
    },
    "nao": {
        "exp_pop_atm1": 6.713,
        "exp_lif_atm1": 5.2861,
        "exp_lix_atm1": 4.896,
        "exp_lifs_sum": 31.9041,
        "exp_lixs_sum": 29.2727,
        "exp_difs_sum": 10.0958,
        "exp_dixs_sum": 12.0802,
        "exp_dif_12": 1.0166,
        "exp_dix_12": 1.2366,
        "exp_iring": 0.254323,
        "exp_mci": 0.373716,
        "exp_av": 47.5604,
        "exp_pdi": 0.248481,
        "Nx": 41.3529
    },
    "iao": {
        "exp_pop_atm1": 6.098,
        "exp_lif_atm1": 4.5235,
        "exp_lix_atm1": 4.1107,
        "exp_lifs_sum": 30.0452,
        "exp_lixs_sum": 27.1053,
        "exp_difs_sum": 11.9548,
        "exp_dixs_sum": 14.2477,
        "exp_dif_12": 1.1222,
        "exp_dix_12": 1.3597,
        "exp_iring": 0.264479,
        "exp_mci": 0.385811,
        "exp_av": 49.2929,
        "exp_pdi": 0.251294,
        "Nx": 41.3529
    }
}

class ESItest(unittest.TestCase):

    def run_all_tests(self, partition, exp):
        esitest = ESI(mol=mol, mf=mf, myhf=myhf, rings=ring, partition=partition)
        aom, occ = esitest.aom
        inds = esitest.indicators[0]

        # 1. Populations and LIs for C1
        pop_atm1 = np.trace(np.dot(occ, aom[0]))
        lif_atm1 = trace(multi_dot((occ ** (1 / 2), aom[0], occ ** (1 / 2), aom[0])))
        lix_atm1 = 0.5 * trace(multi_dot((occ, aom[0], occ, aom[0])))
        
        self.assertAlmostEqual(pop_atm1, exp['exp_pop_atm1'], places=4)
        self.assertAlmostEqual(lif_atm1, exp['exp_lif_atm1'], places=4)
        self.assertAlmostEqual(lix_atm1, exp['exp_lix_atm1'], places=4)

        # 2. DIs
        dif_12 = 2 * trace(multi_dot((occ ** (1 / 2), aom[0], occ ** (1 / 2), aom[1])))
        dix_12 = trace(multi_dot((occ, aom[0], occ, aom[1])))
        self.assertAlmostEqual(dif_12, exp['exp_dif_12'], places=4)
        self.assertAlmostEqual(dix_12, exp['exp_dix_12'], places=4)

        # 3. Sums
        lifs = [trace(multi_dot((occ ** (1 / 2), aom[i], occ ** (1 / 2), aom[i]))) for i in range(len(aom))]
        lixs = [0.5 * trace(multi_dot((occ, aom[i], occ, aom[i]))) for i in range(len(aom))]
        dixs_sum = sum(0.5 * trace(multi_dot((occ, aom[i], occ, aom[j]))) for i in range(len(aom)) for j in range(len(aom)) if i != j)
        
        self.assertAlmostEqual(sum(lifs), exp['exp_lifs_sum'], places=4)
        self.assertAlmostEqual(sum(lixs), exp['exp_lixs_sum'], places=4)
        self.assertAlmostEqual(dixs_sum, exp['exp_dixs_sum'], places=4)
        self.assertAlmostEqual(sum(lixs) + dixs_sum, exp['Nx'], places=4)

        # 4. Indicators
        self.assertAlmostEqual(inds.iring, exp['exp_iring'], places=5)
        self.assertAlmostEqual(inds.mci, exp['exp_mci'], places=5)
        self.assertAlmostEqual(inds.av1245, exp['exp_av'], places=2)
        self.assertAlmostEqual(inds.pdi, exp['exp_pdi'], places=5)

    def test_mulliken(self):
        self.run_all_tests('mulliken', expected['mulliken'])

    def test_lowdin(self):
        self.run_all_tests('lowdin', expected['lowdin'])

    def test_meta_lowdin(self):
        self.run_all_tests('meta-lowdin', expected['meta-lowdin'])

    def test_nao(self):
        self.run_all_tests('nao', expected['nao'])

    def test_iao(self):
        self.run_all_tests('iao', expected['iao'])

if __name__ == '__main__':
    unittest.main()
