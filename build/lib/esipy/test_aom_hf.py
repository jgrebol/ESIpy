import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))
from esipy.readfchk import readfchk
from esipy.make_aoms import make_aoms
import numpy as np

path = '../joan/LiH/GS/lih_1.0.fchk'
mol, mf, myhf = readfchk(path)

print("Running make_aoms with partition='iao' using myhf (HF)")
# To force use of HF, we can pass myhf as the primary mf
# But make_aoms will see it has integer occupations and treat it as RHF
aom_iao_hf = make_aoms(mol, myhf, 'iao')
# For RHF, it returns just the aoms list
pop0_hf = 2 * np.trace(aom_iao_hf[0])
pop1_hf = 2 * np.trace(aom_iao_hf[1])
print(f"IAO HF Populations: {pop0_hf:.4f}, {pop1_hf:.4f}, Total: {pop0_hf+pop1_hf:.4f}")
