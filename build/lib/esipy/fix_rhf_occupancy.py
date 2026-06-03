import os
import re

with open('readfchk.py', 'r') as f:
    content = f.read()

# Restore 2.0 occupancy for RHF fallback
rhf_occ_old = "self.mo_occ[:nocc] = 1.0" # Wait, what was it?
# Let's check the file content first to be sure of the string.
