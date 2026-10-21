"""
example11.py

Reference for IOData:
T. Verstraelen et al., J. Comput. Chem. 2021, 42(6), 458-464.
https://github.com/theochem/iodata
"""

import os
from esipy import ESI, readfchk

molden_file = os.path.join(os.path.dirname(__file__), "..", "tests", "IODATA", "bz.molden")

# Automatic conversion from Molden format to FCHK via IOData
mol, mf = readfchk(molden_file)

# Calculate indicators for benzene
esi = ESI(mol=mol, mf=mf, rings="find", partition="nao")
esi.print()
