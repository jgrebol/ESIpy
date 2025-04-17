import unittest
import os
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

rest = dft.RKS(mol)
rest.xc = 'B3LYP'
rest.kernel()

mol.spin = 2

unrest = dft.UKS(mol)
unrest.xc = 'B3LYP'
unrest.kernel()

ring = [1, 2, 3, 4, 5, 6]


class ESItest(unittest.TestCase):

    def setUp(self):
        # Change to the directory where the script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)

    def test_write_read_aoms_rest(self):
        name = "test4_atomicfiles_rest"
        esitest_wnao = esipy.ESI(mol=mol, mf=rest, rings=ring, partition='nao',
                                 save=name + "_nao")
        esitest_wnao.writeaoms(name + "_nao")
        esitest_wmul = esipy.ESI(mol=mol, mf=rest, rings=ring, partition='m',
                                 save=name + "_mul")
        esitest_wmul.writeaoms(name + "_mul")

        #esitest_rnao = esipy.ESI(read=True, molinfo=name + "_nao.molinfo", rings=ring, partition='nao', name=name,
        #                   readpath=name + '_nao')
        #esitest_rnao.readaoms()
        #esitest_rnao.print()
        #esitest_rmul = esipy.ESI(read=True, molinfo=name + "_mul.molinfo", rings=ring, partition='m', name=name, readpath=name + "_mul")
        #esitest_rmul.readaoms()
        #esitest_rmul.print()

    def test_write_read_aoms_unrest(self):
        name = "test4_atomicfiles_unrest"
        esitest_wnao = esipy.ESI(mol=mol, mf=unrest, rings=ring, partition='nao',
                                 save=name + "_nao")
        esitest_wnao.writeaoms(name + "_nao")
        esitest_wmul = esipy.ESI(mol=mol, mf=unrest, rings=ring, partition='m',
                                 save=name + "_mul")
        esitest_wmul.writeaoms(name + "_mul")
        #esitest_rnao = esipy.ESI(read=True, molinfo=name + "_nao.molinfo", rings=ring, partition='nao', name=name,
        #                   readpath=name + "_nao")
        #esitest_rnao.readaoms()
        #esitest_rnao.print()
        #esitest_rmul = esipy.ESI(read=True, molinfo=name + "_mul.molinfo", rings=ring, partition='m', name=name, readpath=name + "_mul")
        #esitest_rmul.readaoms()
        #esitest_rmul.print()

if __name__ == "__main__":
    unittest.main()
