import unittest
import esipy
from pyscf import gto, dft

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

rest = dft.RKS(mol)
rest.xc = 'B3LYP'
rest.kernel()

mol.spin = 2

unrest = dft.UKS(mol)
unrest.xc = 'B3LYP'
unrest.kernel()

ring = [1, 2, 3, 4, 5, 6]

class ESItest(unittest.TestCase):

    def test_write_aoms_rest(self):
        name = "test4_atomicfiles_rest"
        esitest = esipy.ESI(mol=mol, mf=rest, rings=ring, partition='nao', name=name, savemolinfo=name + "_nao.molinfo")
        esitest.writeaoms()
        esitest = esipy.ESI(mol=mol, mf=rest, rings=ring, partition='m', name=name, savemolinfo=name + "_mul.molinfo")
        esitest.writeaoms()

    def test_read_aoms_rest(self):
        name = "test4_atomicfiles_rest"
        esitest = esipy.ESI(molinfo=name + "_nao.molinfo", rings=ring, partition='nao', name=name, readpath=name + '_nao')
        esitest.readaoms()
        esitest.print()
        esitest = esipy.ESI(molinfo=name + "_mul.molinfo", rings=ring, partition='m', name=name, readpath=name + "_mul")
        esitest.readaoms()
        esitest.print()


    def test_read_aoms_unrest(self):
        name = "test4_atomicfiles_unrest"
        esitest = esipy.ESI(molinfo=name+"_nao.molinfo", rings=ring, partition='nao', name=name, readpath = name +"_nao")
        esitest.readaoms()
        esitest.print()
        esitest = esipy.ESI(molinfo=name+"_mul.molinfo", rings=ring, partition='m', name=name, readpath=name + "_mul")
        esitest.readaoms()
        esitest.print()

    def test_write_aoms_unrest(self):
        name = "test4_atomicfiles_unrest"
        esitest = esipy.ESI(mol=mol, mf=unrest, rings=ring, partition='nao', name=name, savemolinfo=name + "_nao.molinfo")
        esitest.writeaoms()
        esitest = esipy.ESI(mol=mol, mf=unrest, rings=ring, partition='m', name= name, savemolinfo=name + "_mul.molinfo")
        esitest.writeaoms()

if __name__ == '__main__':
    # Create a test suite and add tests in the desired order
    suite = unittest.TestSuite()
    suite.addTest(ESItest('test_write_aoms_rest'))
    suite.addTest(ESItest('test_read_aoms_rest'))
    suite.addTest(ESItest('test_write_aoms_unrest'))
    suite.addTest(ESItest('test_read_aoms_unrest'))

    # Run the test suite
    runner = unittest.TextTestRunner()
    runner.run(suite)

