from pyscf import gto, scf, mp
import traceback
atom = """
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
"""
mol = gto.M(atom=atom, basis='sto-3g', spin=0, charge=0)
myhf = scf.RHF(mol); myhf.init_guess = 'atom'; myhf.kernel()
mf = mp.MP2(myhf); mf.kernel()
try:
    print("Trying make_rdm1(ao_repr=True)")
    d1 = mf.make_rdm1(ao_repr=True)
    print("Success")
except Exception as e:
    print("Failed")
    traceback.print_exc()

