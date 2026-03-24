import unittest
import os
import numpy as np
from esipy.readfchk import readfchk
from esipy import ESI
from pyscf import gto, scf

class TestFchkValidation(unittest.TestCase):
    """
    Comprehensive validation of ESIpy features across sources.
    """

    def setUp(self):
        self.fchk_dir = os.path.join(os.path.dirname(__file__), "..", "FCHK")
        self.systems = {
            "H2O": {
                "g_path": os.path.join(self.fchk_dir, "GAUSSIAN", "h2o.fchk"),
                "q_path": os.path.join(self.fchk_dir, "QCHEM", "h2o.fchk"),
                "rings": None,
                "nelec": 10
            },
            "Benzene": {
                "g_path": os.path.join(self.fchk_dir, "GAUSSIAN", "bz.fchk"),
                "q_path": os.path.join(self.fchk_dir, "QCHEM", "bz.fchk"),
                "rings": [[1, 2, 3, 4, 5, 6]],
                "nelec": 42
            }
        }

    def run_full_comparison(self, system_name, partition):
        sys_info = self.systems[system_name]
        g_path = sys_info["g_path"]
        q_path = sys_info["q_path"]
        
        g_exists = os.path.exists(g_path)
        q_exists = os.path.exists(q_path)

        print(f"\n--- System: {system_name} | Partition: {partition} ---")
        
        results = {}

        # 1. PySCF Reference (from Gaussian geometry)
        if g_exists:
            mol_ref, mf_ref = readfchk(g_path)
            # Rebuild PySCF molecule
            pyscf_mol = gto.M(atom=mol_ref.atom, basis=mol_ref.basis, unit='Bohr', cart=mol_ref.cart)
            pyscf_mf = scf.RHF(pyscf_mol)
            pyscf_mf.kernel()
            results['PySCF'] = ESI(mol=pyscf_mol, mf=pyscf_mf, partition=partition, rings=sys_info["rings"])

        # 2. Gaussian
        if g_exists:
            mol_g, mf_g = readfchk(g_path, is_qchem=False)
            results['Gaussian'] = ESI(mol=mol_g, mf=mf_g, partition=partition, rings=sys_info["rings"])

        # 3. Q-Chem
        if q_exists:
            mol_q, mf_q = readfchk(q_path, is_qchem=True)
            results['QChem'] = ESI(mol=mol_q, mf=mf_q, partition=partition, rings=sys_info["rings"])
            print(f"QChem atom order: {[a[0] for a in mol_q.atom]}")
            # print(f"QChem coords: {mol_q.atom}")

        if g_exists:
            print(f"Gaussian atom order: {[a[0] for a in mol_g.atom]}")
            # print(f"Gaussian coords: {mol_g.atom}")

        if not results:
            self.skipTest(f"No FCHK files found for {system_name}")

        print(f"{'Source':<12} | {'Total e-':<10} | {'Tr(Enter)':<10} | {'LI(Atom1)':<10} | {'MCI':<10}")
        print("-" * 65)

        for source in ['PySCF', 'Gaussian', 'QChem']:
            if source not in results: continue
            esi = results[source]
            
            trace_sum = np.sum([np.trace(m) for m in esi.aom])
            li1 = 2.0 * np.einsum('ij,ji->', esi.aom[0], esi.aom[0])
            mci = esi.indicators[0].mci if esi.rings else 0.0
            
            print(f"{source:<12} | {trace_sum*2.0:<10.5f} | {trace_sum:<10.5f} | {li1:<10.5f} | {mci:<10.5f}")

            # Verification - FCHKs might have small numerical variations in orthogonality
            self.assertAlmostEqual(trace_sum * 2.0, sys_info["nelec"], delta=0.2)
            
            # LI/DI sum consistency
            # Note: For RHF, sum(LI) + sum(DI)/2 = N
            # Our LI = 4*Tr(m^2) and DI = 4*Tr(mi*mj)
            # So sum(LI) + sum_{i<j} DI = sum_{i,j} 2*Tr(mi*mj) * 2 = 2 * N
            # Wait, let's use a more robust check:
            natoms = len(esi.aom)
            li_sum = sum(2.0 * np.einsum('ij,ji->', m, m) for m in esi.aom)
            di_sum = 0.0
            for i in range(natoms):
                for j in range(i + 1, natoms):
                    di_sum += 4.0 * np.einsum('ij,ji->', esi.aom[i], esi.aom[j])
            
            # The sum of populations should be exactly nelec
            pops = [2.0 * np.trace(m) for m in esi.aom]
            self.assertAlmostEqual(sum(pops), sys_info["nelec"], delta=0.1)

        # Cross-source comparisons
        sources = [s for s in ['PySCF', 'Gaussian', 'QChem'] if s in results]
        if len(sources) > 1:
            ref_source = sources[0]
            for other_source in sources[1:]:
                print(f"Comparing {other_source} to {ref_source}...")
                ref_esi = results[ref_source]
                oth_esi = results[other_source]
                
                # 1. Compare atomic populations
                ref_pops = np.array([2.0 * np.trace(m) for m in ref_esi.aom])
                oth_pops = np.array([2.0 * np.trace(m) for m in oth_esi.aom])
                
                np.testing.assert_allclose(ref_pops, oth_pops, atol=1e-2, 
                                           err_msg=f"Populations mismatch between {ref_source} and {other_source}")

                # 2. Compare LI
                ref_li = np.array([2.0 * np.einsum('ij,ji->', m, m) for m in ref_esi.aom])
                oth_li = np.array([2.0 * np.einsum('ij,ji->', m, m) for m in oth_esi.aom])
                np.testing.assert_allclose(ref_li, oth_li, atol=5e-2,
                                           err_msg=f"LI mismatch between {ref_source} and {other_source}")

                # 3. Compare Iring and MCI
                if sys_info["rings"]:
                    for r_idx in range(len(sys_info["rings"])):
                        ref_iring = ref_esi.indicators[r_idx].iring
                        oth_iring = oth_esi.indicators[r_idx].iring
                        ref_mci = ref_esi.indicators[r_idx].mci
                        oth_mci = oth_esi.indicators[r_idx].mci
                        
                        self.assertAlmostEqual(ref_iring, oth_iring, delta=1e-3,
                                               msg=f"Iring mismatch for ring {r_idx} between {ref_source} and {other_source}")
                        self.assertAlmostEqual(ref_mci, oth_mci, delta=1e-3,
                                               msg=f"MCI mismatch for ring {r_idx} between {ref_source} and {other_source}")

    def test_h2o_mulliken(self): self.run_full_comparison("H2O", "mulliken")
    def test_h2o_metalowdin(self): self.run_full_comparison("H2O", "meta-lowdin")
    def test_h2o_iao(self): self.run_full_comparison("H2O", "iao")

    def test_benzene_mulliken(self): self.run_full_comparison("Benzene", "mulliken")
    def test_benzene_metalowdin(self): self.run_full_comparison("Benzene", "meta-lowdin")
    def test_benzene_iao(self): self.run_full_comparison("Benzene", "iao")

if __name__ == "__main__":
    unittest.main()
