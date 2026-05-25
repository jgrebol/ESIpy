import unittest
import os
import numpy as np
import sys

# Ensure esipy is in path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from esipy.readfchk import readfchk
from esipy import ESI

EXPECTED_DATA = {
    ('GAUSSIAN', '10_casscf_rest.fchk'): {'mulliken': {'nelec': 2.0, 'pop_at1': 1.0, 'di12': 0.0223}, 'lowdin': {'nelec': 2.0, 'pop_at1': 1.0, 'di12': 0.0236}, 'nao': {'nelec': 2.0, 'pop_at1': 1.0, 'di12': 0.0216}, 'iao': {'nelec': 2.0, 'pop_at1': 1.0, 'di12': 0.0215}},
    ('GAUSSIAN', '11_ump2.fchk'): {'mulliken': {'nelec': 16.0, 'pop_at1': 8.0, 'di12': 1.2551}, 'lowdin': {'nelec': 16.0, 'pop_at1': 8.0, 'di12': 1.8038}},
    ('GAUSSIAN', '12_casscf_unrest.fchk'): {'mulliken': {'nelec': 16.0, 'pop_at1': 8.0, 'di12': 1.4059}, 'lowdin': {'nelec': 16.0, 'pop_at1': 8.0, 'di12': 1.9159}},
    ('GAUSSIAN', '1_benzene_spherical.fchk'): {'mulliken': {'nelec': 42.0, 'pop_at1': 6.1034, 'di12': 0.6847, 'iring': 0.064726, 'mci': 0.09912}, 'lowdin': {'nelec': 42.0, 'pop_at1': 5.9841, 'di12': 0.7436, 'iring': 0.018489, 'mci': 0.0357}, 'nao': {'nelec': 42.0, 'pop_at1': 6.1939, 'di12': 0.7191, 'iring': 0.086843, 'mci': 0.130048}, 'iao': {'nelec': 42.0, 'pop_at1': 6.1461, 'di12': 0.7193, 'iring': 0.088304, 'mci': 0.132158}},
    ('GAUSSIAN', '2_benzene_cartesian.fchk'): {'mulliken': {'nelec': 42.0, 'pop_at1': 6.1125, 'di12': 0.6905, 'iring': 0.062585, 'mci': 0.098705}, 'lowdin': {'nelec': 42.0, 'pop_at1': 6.0326, 'di12': 0.7612, 'iring': 0.034805, 'mci': 0.054627}, 'nao': {'nelec': 42.0, 'pop_at1': 6.194, 'di12': 0.7195, 'iring': 0.086746, 'mci': 0.129918}, 'iao': {'nelec': 42.0, 'pop_at1': 6.1466, 'di12': 0.7193, 'iring': 0.088304, 'mci': 0.132158}},
    ('GAUSSIAN', '3_o2_triplet.fchk'): {'mulliken': {'nelec': 16.0, 'pop_at1': 8.0, 'di12': 1.3709}, 'lowdin': {'nelec': 16.0, 'pop_at1': 8.0, 'di12': 1.9238}},
    ('GAUSSIAN', '4_h2_oss.fchk'): {'mulliken': {'nelec': 2.0, 'pop_at1': 1.0, 'di12': 0.014}, 'lowdin': {'nelec': 2.0, 'pop_at1': 1.0, 'di12': 0.0143}},
    ('GAUSSIAN', '5_high_l.fchk'): {'mulliken': {'nelec': 10.0, 'pop_at1': 8.4869, 'di12': 0.5223}, 'lowdin': {'nelec': 10.0, 'pop_at1': 7.1489, 'di12': 0.8487}, 'nao': {'nelec': 10.0, 'pop_at1': 8.9322, 'di12': 0.3929}, 'iao': {'nelec': 10.0, 'pop_at1': 8.7304, 'di12': 0.4333}},
    ('GAUSSIAN', '6_ecp.fchk'): {'mulliken': {'nelec': 14.0, 'pop_at1': 7.0, 'di12': 0.4642}, 'lowdin': {'nelec': 14.0, 'pop_at1': 7.0, 'di12': 0.5576}, 'nao': {'nelec': 14.0, 'pop_at1': 7.0, 'di12': 0.6806}, 'iao': {'nelec': 14.0, 'pop_at1': 7.0, 'di12': 0.5166}},
    ('GAUSSIAN', '7_rmp2.fchk'): {'mulliken': {'nelec': 10.0, 'pop_at1': 8.2835, 'di12': 0.9879}, 'lowdin': {'nelec': 10.0, 'pop_at1': 8.0924, 'di12': 1.1513}, 'nao': {'nelec': 10.0, 'pop_at1': 8.8875, 'di12': 0.7596}, 'iao': {'nelec': 9.8529, 'pop_at1': 8.5607, 'di12': 0.8135}},
    ('GAUSSIAN', '8_cisd.fchk'): {'mulliken': {'nelec': 10.0, 'pop_at1': 8.2749, 'di12': 0.9859}, 'lowdin': {'nelec': 10.0, 'pop_at1': 8.0877, 'di12': 1.1498}, 'nao': {'nelec': 10.0, 'pop_at1': 8.8768, 'di12': 0.7618}, 'iao': {'nelec': 9.8546, 'pop_at1': 8.5585, 'di12': 0.8123}},
    ('GAUSSIAN', '9_ccsd.fchk'): {'mulliken': {'nelec': 10.0, 'pop_at1': 8.2711, 'di12': 0.9806}, 'lowdin': {'nelec': 10.0, 'pop_at1': 8.0855, 'di12': 1.145}, 'nao': {'nelec': 10.0, 'pop_at1': 8.8713, 'di12': 0.7575}, 'iao': {'nelec': 9.8481, 'pop_at1': 8.5484, 'di12': 0.8074}},
    ('QCHEM', '10_casscf_rest.fchk'): {'mulliken': {'nelec': 2.0, 'pop_at1': 1.0, 'di12': 0.0223}, 'lowdin': {'nelec': 2.0, 'pop_at1': 1.0, 'di12': 0.0236}, 'nao': {'nelec': 2.0, 'pop_at1': 1.0, 'di12': 0.0216}, 'iao': {'nelec': 2.0, 'pop_at1': 1.0, 'di12': 0.0215}},
    ('QCHEM', '11_ump2.fchk'): {'mulliken': {'nelec': 16.0, 'pop_at1': 8.0, 'di12': 1.4059}, 'lowdin': {'nelec': 16.0, 'pop_at1': 8.0, 'di12': 1.9157}},
    ('QCHEM', '12_casscf_rohf.fchk'): {'mulliken': {'nelec': 18.0, 'pop_at1': 9.0, 'di12': 0.4457}, 'lowdin': {'nelec': 18.0, 'pop_at1': 9.0, 'di12': 0.7732}},
    ('QCHEM', '1_benzene_spherical.fchk'): {'mulliken': {'nelec': 42.0, 'pop_at1': 6.1038, 'di12': 0.6849, 'iring': 0.064701, 'mci': 0.099085}, 'lowdin': {'nelec': 42.0, 'pop_at1': 5.9841, 'di12': 0.7435, 'iring': 0.018487, 'mci': 0.035698}, 'nao': {'nelec': 42.0, 'pop_at1': 6.1939, 'di12': 0.7192, 'iring': 0.086842, 'mci': 0.130047}, 'iao': {'nelec': 42.0, 'pop_at1': 6.1462, 'di12': 0.7194, 'iring': 0.088304, 'mci': 0.132158}},
    ('QCHEM', '2_benzene_cartesian.fchk'): {'mulliken': {'nelec': 42.0, 'pop_at1': 6.113, 'di12': 0.6908, 'iring': 0.062564, 'mci': 0.098671}, 'lowdin': {'nelec': 42.0, 'pop_at1': 6.0327, 'di12': 0.7612, 'iring': 0.034802, 'mci': 0.054623}, 'nao': {'nelec': 42.0, 'pop_at1': 6.194, 'di12': 0.7195, 'iring': 0.086745, 'mci': 0.129916}, 'iao': {'nelec': 42.0, 'pop_at1': 6.1466, 'di12': 0.7194, 'iring': 0.088304, 'mci': 0.132158}},
    ('QCHEM', '3_o2_triplet.fchk'): {'mulliken': {'nelec': 16.0, 'pop_at1': 8.0, 'di12': 1.3709}, 'lowdin': {'nelec': 16.0, 'pop_at1': 8.0, 'di12': 1.9238}},
    ('QCHEM', '4_h2_oss.fchk'): {'mulliken': {'nelec': 2.0, 'pop_at1': 1.0, 'di12': 0.014}, 'lowdin': {'nelec': 2.0, 'pop_at1': 1.0, 'di12': 0.0143}},
    ('QCHEM', '5_high_l.fchk'): {'mulliken': {'nelec': 10.0, 'pop_at1': 8.4869, 'di12': 0.5222}, 'lowdin': {'nelec': 10.0, 'pop_at1': 7.1489, 'di12': 0.8487}, 'nao': {'nelec': 10.0, 'pop_at1': 8.9322, 'di12': 0.3929}, 'iao': {'nelec': 10.0, 'pop_at1': 8.7304, 'di12': 0.4333}},
    ('QCHEM', '6_ecp.fchk'): {'mulliken': {'nelec': 14.0, 'pop_at1': 7.0, 'di12': 0.4643}, 'lowdin': {'nelec': 14.0, 'pop_at1': 7.0, 'di12': 0.5576}, 'nao': {'nelec': 14.0, 'pop_at1': 7.0, 'di12': 0.6806}, 'iao': {'nelec': 14.0, 'pop_at1': 7.0, 'di12': 0.5166}},
    ('QCHEM', '7_rmp2.fchk'): {'mulliken': {'nelec': 10.0, 'pop_at1': 8.3056, 'di12': 0.5103}, 'lowdin': {'nelec': 10.0, 'pop_at1': 8.1049, 'di12': 0.5915}, 'nao': {'nelec': 10.0, 'pop_at1': 8.9175, 'di12': 0.3956}, 'iao': {'nelec': 10.0, 'pop_at1': 8.7023, 'di12': 0.4383}},
    ('QCHEM', '8_cisd.fchk'): {'mulliken': {'nelec': 10.0, 'pop_at1': 8.3056, 'di12': 0.5103}, 'lowdin': {'nelec': 10.0, 'pop_at1': 8.1049, 'di12': 0.5915}, 'nao': {'nelec': 10.0, 'pop_at1': 8.9175, 'di12': 0.3956}, 'iao': {'nelec': 10.0, 'pop_at1': 8.7023, 'di12': 0.4383}},
    ('QCHEM', '9_ccsd.fchk'): {'mulliken': {'nelec': 10.0, 'pop_at1': 8.2701, 'di12': 0.9813}, 'lowdin': {'nelec': 10.0, 'pop_at1': 8.0852, 'di12': 1.1455}, 'nao': {'nelec': 10.0, 'pop_at1': 8.8712, 'di12': 0.7576}, 'iao': {'nelec': 9.8485, 'pop_at1': 8.5478, 'di12': 0.8078}},
}

class TestFchkValidation(unittest.TestCase):
    def setUp(self):
        self.base_dir = os.path.join(os.path.dirname(__file__), 'FCHK')
        self.partitions = ['mulliken', 'lowdin', 'nao', 'iao']

    def get_pops(self, esi):
        from esipy.tools import wf_type
        aoms = esi.aom
        wf = wf_type(aoms)
        if wf == "rest":
            return np.array([2 * np.trace(m) for m in aoms])
        elif wf == "unrest":
            return np.array([np.trace(a) + np.trace(b) for a, b in zip(aoms[0], aoms[1])])
        elif wf == "no":
            aoms_list, occ_matrix = aoms
            if occ_matrix.ndim == 1:
                return np.array([np.einsum('i,ii->', occ_matrix, m) for m in aoms_list])
            else:
                return np.array([np.trace(occ_matrix @ m) for m in aoms_list])
        return np.array([])

    def run_validation(self, source):
        for (prog, filename), expected_sys in EXPECTED_DATA.items():
            if prog != source: continue
            path = os.path.join(self.base_dir, source, filename)
            if not os.path.exists(path): continue
            
            with self.subTest(system=filename, program=source):
                mol, mf = readfchk(path)
                rings = [[1,2,3,4,5,6]] if 'benzene' in filename else None

                for p in self.partitions:
                    if p not in expected_sys: continue
                    expected = expected_sys[p]
                    
                    with self.subTest(partition=p):
                        esi = ESI(mol=mol, mf=mf, partition=p, rings=rings)
                        pops = self.get_pops(esi)
                        pop_sum = np.sum(pops)
                        
                        self.assertAlmostEqual(pop_sum, expected['nelec'], delta=0.01,
                                         msg=f"Pop sum mismatch in {filename} ({p})")
                        
                        self.assertAlmostEqual(pops[0], expected['pop_at1'], delta=0.01,
                                         msg=f"Pop atom 1 mismatch in {filename} ({p})")

                        if 'di12' in expected:
                            from esipy.tools import wf_type, find_di, find_di_no
                            wf = wf_type(esi.aom)
                            if wf == "rest":
                                di12 = find_di(esi.aom, 1, 2)
                            elif wf == "unrest":
                                di12 = 2 * (find_di(esi.aom[0], 1, 2) + find_di(esi.aom[1], 1, 2))
                            else:
                                di12 = find_di_no(esi.aom, 1, 2)
                            self.assertAlmostEqual(di12, expected['di12'], delta=0.01,
                                             msg=f"DI12 mismatch in {filename} ({p})")

                        if rings and 'iring' in expected:
                             self.assertAlmostEqual(esi.indicators[0].iring, expected['iring'], delta=0.001)
                             self.assertAlmostEqual(esi.indicators[0].mci, expected['mci'], delta=0.001)

    def test_gaussian(self):
        self.run_validation('GAUSSIAN')

    def test_qchem(self):
        self.run_validation('QCHEM')

if __name__ == '__main__':
    unittest.main()
