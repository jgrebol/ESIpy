import unittest
import os
import numpy as np
from esipy.readfchk import readfchk
from esipy import ESI

EXPECTED_DATA = {
    ('GAUSSIAN', '10_casscf_rest.fchk'): {'mulliken': {'nelec': 4.0, 'pop_at1': 2.0}, 'lowdin': {'nelec': 4.0, 'pop_at1': 2.0}, 'nao': {'nelec': 4.0, 'pop_at1': 2.0}, 'iao': {'nelec': 4.0, 'pop_at1': 2.0}},
    ('GAUSSIAN', '11_ump2.fchk'): {'mulliken': {'nelec': 18.0, 'pop_at1': 9.0}, 'lowdin': {'nelec': 18.0, 'pop_at1': 9.0}},
    ('GAUSSIAN', '12_casscf_unrest.fchk'): {'mulliken': {'nelec': 18.0, 'pop_at1': 9.0}, 'lowdin': {'nelec': 18.0, 'pop_at1': 9.0}},
    ('GAUSSIAN', '3_o2_triplet.fchk'): {'mulliken': {'nelec': 18.0, 'pop_at1': 9.0}, 'lowdin': {'nelec': 18.0, 'pop_at1': 9.0}},
    ('GAUSSIAN', '4_h2_oss.fchk'): {'mulliken': {'nelec': 4.0, 'pop_at1': 2.0}, 'lowdin': {'nelec': 4.0, 'pop_at1': 2.0}},
    ('GAUSSIAN', '7_rmp2.fchk'): {'mulliken': {'nelec': 10.0, 'pop_at1': 8.2875, 'di12': 0.5148}, 'lowdin': {'nelec': 10.0, 'pop_at1': 8.0978, 'di12': 0.5946}, 'nao': {'nelec': 10.0, 'pop_at1': 8.9100}, 'iao': {'nelec': 10.0, 'pop_at1': 8.6947}},
    ('GAUSSIAN', '8_cisd.fchk'): {'mulliken': {'nelec': 10.0, 'pop_at1': 8.2807, 'di12': 0.5147}, 'lowdin': {'nelec': 10.0, 'pop_at1': 8.0945, 'di12': 0.5943}, 'nao': {'nelec': 10.0, 'pop_at1': 8.9008}, 'iao': {'nelec': 10.0, 'pop_at1': 8.6905}},
    ('GAUSSIAN', '9_ccsd.fchk'): {'mulliken': {'nelec': 10.0, 'pop_at1': 8.2768, 'di12': 0.5151}, 'lowdin': {'nelec': 10.0, 'pop_at1': 8.0931, 'di12': 0.5946}, 'nao': {'nelec': 10.0, 'pop_at1': 8.8986}, 'iao': {'nelec': 10.0, 'pop_at1': 8.6886}},
    ('QCHEM', '10_casscf_rest.fchk'): {'mulliken': {'nelec': 4.0, 'pop_at1': 2.0}, 'lowdin': {'nelec': 4.0, 'pop_at1': 2.0}, 'nao': {'nelec': 4.0, 'pop_at1': 2.0}, 'iao': {'nelec': 4.0, 'pop_at1': 2.0}},
    ('QCHEM', '11_ump2.fchk'): {'mulliken': {'nelec': 18.0, 'pop_at1': 9.0}, 'lowdin': {'nelec': 18.0, 'pop_at1': 9.0}},
    ('QCHEM', '3_o2_triplet.fchk'): {'mulliken': {'nelec': 18.0, 'pop_at1': 9.0}, 'lowdin': {'nelec': 18.0, 'pop_at1': 9.0}},
    ('QCHEM', '4_h2_oss.fchk'): {'mulliken': {'nelec': 4.0, 'pop_at1': 2.0}, 'lowdin': {'nelec': 4.0, 'pop_at1': 2.0}},
    ('QCHEM', '7_rmp2.fchk'): {'mulliken': {'nelec': 10.0, 'pop_at1': 8.3056, 'di12': 0.5103}, 'lowdin': {'nelec': 10.0, 'pop_at1': 8.1049, 'di12': 0.5915}, 'nao': {'nelec': 10.0, 'pop_at1': 8.9175}, 'iao': {'nelec': 10.0, 'pop_at1': 8.7023}},
    ('QCHEM', '8_cisd.fchk'): {'mulliken': {'nelec': 10.0, 'pop_at1': 8.3056, 'di12': 0.5103}, 'lowdin': {'nelec': 10.0, 'pop_at1': 8.1049, 'di12': 0.5915}, 'nao': {'nelec': 10.0, 'pop_at1': 8.9175}, 'iao': {'nelec': 10.0, 'pop_at1': 8.7023}},
    ('QCHEM', '9_ccsd.fchk'): {'mulliken': {'nelec': 10.0, 'pop_at1': 8.2768, 'di12': 0.5155}, 'lowdin': {'nelec': 10.0, 'pop_at1': 8.0931, 'di12': 0.5948}, 'nao': {'nelec': 10.0, 'pop_at1': 8.8986}, 'iao': {'nelec': 10.0, 'pop_at1': 8.6879}},
}

class TestFchkValidation(unittest.TestCase):
    def setUp(self):
        self.base_dir = os.path.join(os.path.dirname(__file__), 'FCHK')
        self.partitions = ['mulliken', 'lowdin', 'nao', 'iao']

    def get_pops(self, esi):
        aoms = esi.aom
        # NOs return format [list, array]
        if isinstance(aoms, list) and len(aoms) == 2 and isinstance(aoms[1], np.ndarray):
            aom_list, occ = aoms
            if occ.ndim == 1:
                occ = np.diag(occ)
            return np.array([np.trace(occ @ m) for m in aom_list])
        # Unrestricted: [list of lists of matrices]
        elif isinstance(aoms, list) and len(aoms) == 2 and isinstance(aoms[0], list):
            return np.array([np.trace(a) + np.trace(b) for a, b in zip(aoms[0], aoms[1])])
        # Default restricted
        return np.array([2 * np.trace(m) for m in aoms])

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
                        
                        self.assertAlmostEqual(pop_sum, expected['nelec'], delta=0.1,
                                         msg=f"Pop sum mismatch in {filename} ({p})")
                        
                        self.assertAlmostEqual(pops[0], expected['pop_at1'], delta=0.1,
                                         msg=f"Pop atom 1 mismatch in {filename} ({p})")

                        if 'di12' in expected:
                            from esipy.tools import wf_type, find_di, find_di_no
                            wf = wf_type(esi.aom)
                            if wf == "no":
                                di12 = find_di_no(esi.aom, 1, 2)
                            elif wf == "unrest":
                                di12 = 2 * (find_di(esi.aom[0], 1, 2) + find_di(esi.aom[1], 1, 2))
                            else:
                                di12 = find_di(esi.aom, 1, 2)
                            self.assertAlmostEqual(di12, expected['di12'], delta=0.1,
                                             msg=f"DI12 mismatch in {filename} ({p})")

    def test_gaussian(self):
        self.run_validation('GAUSSIAN')

    def test_qchem(self):
        self.run_validation('QCHEM')

if __name__ == '__main__':
    unittest.main()
