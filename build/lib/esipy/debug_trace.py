import os
import sys
import numpy as np
import traceback

sys.path.append(os.getcwd())
from esipy.readfchk import readfchk
from esipy import ESI

path = '../tests/FCHK/QCHEM/h2o_sto3g.fchk'
try:
    mol, mf = readfchk(path)
    print("Success")
except Exception as e:
    traceback.print_exc()
