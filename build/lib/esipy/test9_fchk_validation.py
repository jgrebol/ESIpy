import unittest
import os
import numpy as np
from esipy.readfchk import readfchk
from esipy import ESI

class TestFchkValidation(unittest.TestCase):
    """
    Comprehensive validation of ESIpy's FCHK reading and AOM building.
    Tests a variety of basis sets, spherical/cartesian functions, and restricted/unrestricted wavefunctions.
    """

    def setUp(self):
        # Determine the base directory for tests and joan data
        self.test_file_path = os.path.abspath(__file__)
        self.test_dir = os.path.dirname(self.test_file_path)
        self.project_root = os.path.dirname(self.test_dir)
        
        # Standard test FCHKs provided with the repository
        self.fchk_dir = os.path.join(self.test_dir, 'FCHK', 'GAUSSIAN')
        self.qchem_fchk_dir = os.path.join(self.test_dir, 'FCHK', 'QCHEM')
        
        # Joan's benchmarking FCHKs (assumed to be in a sibling 'joan' directory)
        self.joan_base = os.path.join(self.project_root, 'joan')

        # Define systems to test
        self.systems = [
            # --- WATER (H2O) - nelec = 10 ---
            {'name': 'H2O_STO3G', 'path': os.path.join(self.fchk_dir, 'h2o_sto3g.fchk'), 'nelec': 10, 'rings': None},
            {'name': 'H2O_631Gss', 'path': os.path.join(self.fchk_dir, 'h2o_631gss.fchk'), 'nelec': 10, 'rings': None},
            {'name': 'H2O_ccpVDZ_cart', 'path': os.path.join(self.fchk_dir, 'h2o_ccpvdz_cart.fchk'), 'nelec': 10, 'rings': None},
            {'name': 'H2O_ccpVDZ_sph', 'path': os.path.join(self.fchk_dir, 'h2o_ccpvdz_sph.fchk'), 'nelec': 10, 'rings': None},
            {'name': 'H2O_ccpVTZ_cart', 'path': os.path.join(self.fchk_dir, 'h2o_ccpvtz_cart.fchk'), 'nelec': 10, 'rings': None},
            {'name': 'H2O_ccpVTZ_sph', 'path': os.path.join(self.fchk_dir, 'h2o_ccpvtz_sph.fchk'), 'nelec': 10, 'rings': None},
            
            # --- Q-Chem WATER (H2O) ---
            {'name': 'H2O_STO3G_QChem', 'path': os.path.join(self.qchem_fchk_dir, 'h2o_sto3g.fchk'), 'nelec': 10, 'rings': None},
            {'name': 'H2O_631Gss_QChem', 'path': os.path.join(self.qchem_fchk_dir, 'h2o_631gss.fchk'), 'nelec': 10, 'rings': None},
            {'name': 'H2O_ccpVDZ_cart_QChem', 'path': os.path.join(self.qchem_fchk_dir, 'h2o_ccpvdz_cart.fchk'), 'nelec': 10, 'rings': None},
            {'name': 'H2O_ccpVDZ_sph_QChem', 'path': os.path.join(self.qchem_fchk_dir, 'h2o_ccpvdz_sph.fchk'), 'nelec': 10, 'rings': None},
            {'name': 'H2O_ccpVTZ_cart_QChem', 'path': os.path.join(self.qchem_fchk_dir, 'h2o_ccpvtz_cart.fchk'), 'nelec': 10, 'rings': None},
            {'name': 'H2O_ccpVTZ_sph_QChem', 'path': os.path.join(self.qchem_fchk_dir, 'h2o_ccpvtz_sph.fchk'), 'nelec': 10, 'rings': None},
            {'name': 'H2O_ccpVQZ_cart_QChem', 'path': os.path.join(self.qchem_fchk_dir, 'h2o_ccpvqz_cart.fchk'), 'nelec': 10, 'rings': None},
            {'name': 'H2O_ccpVQZ_sph_QChem', 'path': os.path.join(self.qchem_fchk_dir, 'h2o_ccpvqz_sph.fchk'), 'nelec': 10, 'rings': None},

            # Using valid files from joan for VQZ (the ones in FCHK/GAUSSIAN/ are placeholders)
            {'name': 'H2O_ccpVQZ', 'path': os.path.join(self.joan_base, 'NEWIAOS', 'H2O', 'FCHK', 'h2o_cc-pVQZ.fchk'), 'nelec': 10, 'rings': None},
            {'name': 'H2O_aug_ccpVQZ', 'path': os.path.join(self.joan_base, 'NEWIAOS', 'H2O', 'FCHK', 'h2o_aug-cc-pVQZ.fchk'), 'nelec': 10, 'rings': None},

            # --- BENZENE (C6H6) - nelec = 42 ---
            # Using automatic ring finder ('f') because atom order varies between files
            {'name': 'Benzene_631Gss', 'path': os.path.join(self.joan_base, 'T01-BSC', 'bz_6-31Gss.fchk'), 'nelec': 42, 'rings': 'f'},
            {'name': 'Benzene_ccpVDZ', 'path': os.path.join(self.joan_base, 'T01-BSC', 'bz_cc-pVDZ.fchk'), 'nelec': 42, 'rings': 'f'},
            {'name': 'Benzene_aug_ccpVQZ', 'path': os.path.join(self.joan_base, 'T01-BSC', 'bz_aug-cc-pVQZ.fchk'), 'nelec': 42, 'rings': 'f'},

            # --- CYCLOBUTADIENE (C4H4) - nelec = 28 ---
            {'name': 'C4H4_Restricted', 'path': os.path.join(self.joan_base, 'c4h4_symm', 'rest.fchk'), 'nelec': 28, 'rings': 'f'},
            {'name': 'C4H4_Unrestricted', 'path': os.path.join(self.joan_base, 'c4h4_symm', 'unrest.fchk'), 'nelec': 28, 'rings': 'f'},

            # --- HYDROGEN (H2) - nelec = 2 ---
            {'name': 'H2_Unrestricted', 'path': os.path.join(self.fchk_dir, 'unrest.fchk'), 'nelec': 2, 'rings': None},
        ]

    def run_system_validation(self, system, partition):
        path = system['path']
        if not os.path.exists(path):
            self.skipTest(f"FCHK file not found: {path}")

        # 1. Read FCHK
        mol, mf = readfchk(path)
        
        # 2. Build AOMs and Indicators
        # rings='f' tells ESI to find rings automatically
        # minlen=3 ensures we find small rings like C4H4
        esi = ESI(mol=mol, mf=mf, partition=partition, rings=system['rings'], minlen=3)
        
        # 3. Validate Population Conservation (Trace Sum)
        # For Restricted: Sum(Tr(AOM)) = Nelec / 2
        # For Unrestricted: Sum(Tr(AOM_alpha)) + Sum(Tr(AOM_beta)) = Nelec
        if isinstance(esi.aom, list) and len(esi.aom) > 0 and isinstance(esi.aom[0], list):
            # Unrestricted case
            trace_sum = sum(np.trace(m) for sub in esi.aom for m in sub)
            expected = float(system['nelec'])
        else:
            # Restricted case
            trace_sum = sum(np.trace(m) for m in esi.aom)
            expected = system['nelec'] / 2.0
            
        self.assertAlmostEqual(trace_sum, expected, delta=0.01, 
                               msg=f"Trace sum failed for {system['name']} with {partition}. "
                                   f"Got {trace_sum}, expected {expected}")

        # 4. Basic Aromaticity Check
        if system['rings'] == 'f':
            # Check if rings were found
            self.assertIsNotNone(esi.rings, msg=f"No rings found for {system['name']}")
            self.assertTrue(hasattr(esi, 'indicators'), msg=f"Indicators not created for {system['name']}")
            
            for ind in esi.indicators:
                # Iring should be non-negative for these stable systems
                self.assertGreater(ind.iring, -1e-4, msg=f"Iring is significantly negative for {system['name']} ({ind.iring})")
                # If it's a known aromatic system (Benzene), Iring should be significant
                if 'Benzene' in system['name']:
                    self.assertGreater(ind.iring, 0.01, msg=f"Iring too low for Benzene in {system['name']} ({ind.iring})")

    def test_iao_validation(self):
        """Test IAO partition for all systems."""
        for sys in self.systems:
            with self.subTest(system=sys['name'], partition='iao'):
                self.run_system_validation(sys, 'iao')

    def test_nao_validation(self):
        """Test NAO partition for a representative subset of systems."""
        subset = [s for s in self.systems if 'STO3G' in s['name'] or 'Unrestricted' in s['name'] or 'Restricted' in s['name']]
        for sys in subset:
            with self.subTest(system=sys['name'], partition='nao'):
                self.run_system_validation(sys, 'nao')

    def test_mulliken_validation(self):
        """Test Mulliken partition for a representative subset of systems."""
        subset = [s for s in self.systems if '631Gss' in s['name']]
        for sys in subset:
            with self.subTest(system=sys['name'], partition='mulliken'):
                self.run_system_validation(sys, 'mulliken')

    def test_lowdin_validation(self):
        """Test Lowdin partition for a representative subset of systems."""
        subset = [s for s in self.systems if 'ccpVDZ' in s['name']]
        for sys in subset:
            with self.subTest(system=sys['name'], partition='lowdin'):
                self.run_system_validation(sys, 'lowdin')

    def test_gaussian_vs_qchem_comparison(self):
        """Compare results between Gaussian and Q-Chem for the same systems."""
        comparison_systems = [
            'H2O_STO3G', 'H2O_631Gss', 'H2O_ccpVDZ_cart', 'H2O_ccpVDZ_sph',
            'H2O_ccpVTZ_cart', 'H2O_ccpVTZ_sph'
        ]
        
        # Add cc-pVQZ if Gaussian files are available in joan folder
        if os.path.exists(os.path.join(self.joan_base, 'NEWIAOS', 'H2O', 'FCHK', 'h2o_cc-pVQZ.fchk')):
            comparison_systems.append('H2O_ccpVQZ')

        for sys_name in comparison_systems:
            with self.subTest(system=sys_name):
                # Gaussian Path
                g_sys = next(s for s in self.systems if s['name'] == sys_name)
                g_path = g_sys['path']
                
                # Q-Chem Path
                basename = os.path.basename(g_path)
                q_path = os.path.join(self.test_dir, 'FCHK', 'QCHEM', basename)
                
                # Special handling for cc-pVQZ mapping if basename didn't work
                if not os.path.exists(q_path):
                    if 'cc-pVQZ' in basename:
                        q_path = os.path.join(self.test_dir, 'FCHK', 'QCHEM', 'h2o_ccpvqz_cart.fchk')
                    elif 'ccpVQZ' in sys_name:
                         q_path = os.path.join(self.test_dir, 'FCHK', 'QCHEM', 'h2o_ccpvqz_cart.fchk')

                if not os.path.exists(g_path) or not os.path.exists(q_path):
                    self.skipTest(f"Missing FCHK for comparison: {sys_name} (G: {os.path.exists(g_path)}, Q: {os.path.exists(q_path)})")
                
                # Read and compute for Gaussian
                g_mol, g_mf = readfchk(g_path)
                g_esi = ESI(mol=g_mol, mf=g_mf, partition='iao')
                g_traces = [np.trace(m) for m in g_esi.aom]
                
                # Read and compute for Q-Chem
                q_mol, q_mf = readfchk(q_path)
                q_esi = ESI(mol=q_mol, mf=q_mf, partition='iao')
                q_traces = [np.trace(m) for m in q_esi.aom]
                
                np.testing.assert_allclose(g_traces, q_traces, atol=2e-3, 
                                           err_msg=f"AOM trace mismatch for {sys_name}")


if __name__ == '__main__':
    unittest.main()
