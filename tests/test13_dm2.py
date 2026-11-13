import time
import unittest

import numpy as np
from pyscf import gto, scf, mcscf, ci, cc, mp

from esipy import ESI
from esipy.tools import make_dm2, dm2_energy, get_natorbs

water = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'
mol = gto.M(atom=water, basis='6-31g', verbose=0)
rhf = scf.RHF(mol).run()
cas = mcscf.CASSCF(rhf, 4, 4).run()
cisd = ci.CISD(rhf).run()

partitions = ['lowdin', 'meta-lowdin', 'nao', 'iao', 'iao-effao-nao']


class NatorbRHF(scf.hf.RHF):
    """RHF orbitals passed as natural orbitals (occupations 2 and 0)."""
    is_natorb = True


def hf_dm2(nocc, nmo):
    """DM2 of a closed-shell determinant in the MO basis: dm2_pqrs = D_pq D_rs - D_ps D_rq / 2."""
    D = np.diag([2.0] * nocc + [0.0] * (nmo - nocc))
    return np.einsum('pq,rs->pqrs', D, D) - 0.5 * np.einsum('ps,rq->pqrs', D, D)


def dm1_mo(mf, mo):
    """DM1 of 'mf' in the basis of the orbitals mo (from its natural orbitals)."""
    S = mf.mol.intor_symmetric('int1e_ovlp')
    occ, coeff = get_natorbs(mf, S)
    U = mo.T @ S @ coeff
    return (U * occ) @ U.T


def sum_rule(esi):
    """max |LI_A + sum_B DI_AB / 2 - N_A|."""
    mat = esi.di_matrix
    aom, occ = esi.aom
    N = np.array([np.dot(occ, np.diag(s)) for s in aom])
    return np.abs(np.diag(mat) + (mat.sum(axis=1) - np.diag(mat)) / 2 - N).max()


def di_exchange(esi):
    """DI_AB from the exchange-only approximation of the DM2 (n_p n_q instead of sqrt(n_p n_q))."""
    aom, occ = esi.aom
    return np.sum(np.outer(occ, occ) * aom[0] * aom[1].T)


def dissociation(atom, basis):
    """Exact (FCI), Fermi and exchange-only DIs of a diatomic, and the single-determinant DI."""
    m = gto.M(atom=atom, basis=basis, verbose=0)
    hf = scf.RHF(m).run()
    fci = mcscf.CASCI(hf, m.nao, m.nelectron).run()
    esi = ESI(mol=m, mf=fci, partition='nao', dm2=True)
    return dict(exact=esi.di_matrix[0, 1], fermi=ESI(mol=m, mf=fci, partition='nao').di_matrix[0, 1],
                exchange=di_exchange(esi), hf=ESI(mol=m, mf=hf, partition='nao').di_matrix[0, 1], sum_rule=sum_rule(esi))


class DM2Test(unittest.TestCase):

    def test_hf_dm2_is_single_determinant(self):
        # The DM2 of the determinant gives the same LIs and DIs as the single-determinant code, and so does the
        # Fermi approximation (occupations 0 and 2). The IAOs span the occupied orbitals in both cases.
        no = NatorbRHF(mol)
        no.mo_coeff, no.mo_occ, no.e_tot = rhf.mo_coeff, rhf.mo_occ, rhf.e_tot
        dm2 = hf_dm2(mol.nelectron // 2, mol.nao)
        self.assertAlmostEqual(dm2_energy(mol, rhf.mo_coeff, dm2), rhf.e_tot, places=8)
        for part in partitions:
            ref = ESI(mol=mol, mf=rhf, partition=part).di_matrix
            np.testing.assert_allclose(ESI(mol=mol, mf=no, partition=part, dm2=dm2).di_matrix, ref, atol=1e-8, err_msg=part)
            np.testing.assert_allclose(ESI(mol=mol, mf=no, partition=part).di_matrix, ref, atol=1e-8, err_msg=part)

    def test_casscf_energy_and_sum_rule(self):
        dm2 = make_dm2(cas)
        self.assertAlmostEqual(dm2_energy(mol, cas.mo_coeff, dm2), cas.e_tot, places=8)
        for part in ('lowdin', 'meta-lowdin', 'nao'):
            esi = ESI(mol=mol, mf=cas, partition=part, dm2=True)
            mat = esi.di_matrix
            aom, occ = esi.aom
            N = [np.dot(occ, np.diag(s)) for s in aom]
            # Exact sum rule with all natural orbitals: LI_A + sum_B DI_AB / 2 = N_A
            np.testing.assert_allclose(np.diag(mat) + (mat.sum(axis=1) - np.diag(mat)) / 2, N, atol=1e-8, err_msg=part)
            np.testing.assert_allclose(mat, mat.T, atol=1e-10)

    def test_dm2_array_equals_dm2_true(self):
        for mf, dm2 in ((cas, make_dm2(cas)), (cisd, cisd.make_rdm2())):
            for part in ('nao', 'iao-effao-nao'):
                a = ESI(mol=mol, mf=mf, partition=part, dm2=True).di_matrix
                b = ESI(mol=mol, mf=mf, partition=part, dm2=dm2).di_matrix
                np.testing.assert_allclose(a, b, atol=1e-12, err_msg=part)

    def test_cisd_energy(self):
        self.assertAlmostEqual(dm2_energy(mol, rhf.mo_coeff, cisd.make_rdm2()), cisd.e_tot, places=8)

    def test_truncated_nocc_conserves_electrons(self):
        # With the IAOs built from the first nocc natural orbitals, the populations add up to their occupations
        for nocc in (5, 6, 7):
            esi = ESI(mol=mol, mf=cas, partition='iao-effao-nao', dm2=True, nocc=nocc)
            aom, occ = esi.aom
            self.assertAlmostEqual(sum(np.dot(occ, np.diag(s)) for s in aom), np.sum(occ[:nocc]), places=8)
            self.assertEqual(esi.di_matrix.shape, (mol.natm, mol.natm))

    def test_all_methods(self):
        # From converged calculations: the DM2 reproduces the energy with the DM1 of each method. The sum rule is exact
        # for the variational (N-representable) wavefunctions; MP2 and CCSD DM2s do not trace exactly to their DM1.
        methods = {'MP2': mp.MP2(rhf).run(), 'CCSD': cc.CCSD(rhf).run(), 'CISD': cisd,
                   'CASCI': mcscf.CASCI(rhf, 6, 6).run(), 'CASSCF': cas, 'FCI': mcscf.CASCI(rhf, mol.nao, mol.nelectron).run()}
        for name, mf in methods.items():
            t = time.time()
            esi = ESI(mol=mol, mf=mf, partition='nao', dm2=True)
            mat = esi.di_matrix
            t = time.time() - t
            mo = mf.mo_coeff[:, :len(esi.dm2)]
            self.assertAlmostEqual(dm2_energy(mol, mo, esi.dm2, dm1_mo(mf, mo)), mf.e_tot, places=6, msg=name)
            if name not in ('MP2', 'CCSD'):
                self.assertLess(sum_rule(esi), 1e-8, msg=name)
            self.assertLess(t, 30, msg=name)
            print(f' | {name}: DM2 {esi.dm2.shape[0]} orbitals, DI(O,H) {mat[0, 1]:.4f}, {t:.2f} s')

    def test_h2_dissociation(self):
        # Static correlation: the exact DI goes to zero, the single determinant keeps DI = 1 and the exchange-only
        # approximation overestimates it
        eq, st = dissociation('H 0 0 0; H 0 0 0.74', 'cc-pvdz'), dissociation('H 0 0 0; H 0 0 4.0', 'cc-pvdz')
        for x in (eq, st):
            self.assertLess(x['sum_rule'], 1e-8)
            self.assertAlmostEqual(x['hf'], 1.0, places=6)
            self.assertGreater(x['exchange'], x['exact'])
        self.assertTrue(0.7 < eq['exact'] < 0.9)
        self.assertLess(st['exact'], 1e-2)

    def test_he2_dimer(self):
        # Closed shells without bond: every DI is close to zero
        x = dissociation('He 0 0 0; He 0 0 3.0', 'cc-pvdz')
        self.assertLess(x['sum_rule'], 1e-8)
        for k in ('exact', 'fermi', 'exchange', 'hf'):
            self.assertLess(abs(x[k]), 1e-3, msg=k)


if __name__ == '__main__':
    unittest.main()
