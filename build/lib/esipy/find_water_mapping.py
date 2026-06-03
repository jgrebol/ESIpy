import numpy as np
from pyscf import gto, scf
from esipy.readfchk import readfchk
import os

def check(prog):
    path = f"../tests/FCHK/{prog}/7_rmp2.fchk"
    if not os.path.exists(path): return
    mol_f, mf_f = readfchk(path)
    S_f = mf_f.get_ovlp()
    
    basis = 'cc-pVDZ'
    atom = "O 0.0 0.0 0.0; H 0.0 0.757 0.586; H 0.0 -0.757 0.586"
    mol_p = gto.M(atom=atom, basis=basis).build()
    S_p = mol_p.intor('int1e_ovlp')
    
    print(f"\n--- WATER {prog} Analysis ---")
    # Oxygen in cc-pVDZ: s(3), p(2), d(1) -> 3+6+5 = 14 AOs
    # Hydrogen: s(2), p(1) -> 2+3 = 5 AOs
    # Total: 14 + 5 + 5 = 24 AOs.
    
    mapping = []
    for i in range(len(S_p)):
        for j in range(len(S_f)):
            if np.allclose(S_p[i, :], S_f[j, :], atol=1e-5):
                mapping.append(j)
                break
                
    if len(mapping) == 24:
        print(f"  FULL MAPPING FOUND: {mapping}")
        # Identify shell mappings
        print(f"  O s-shells: {mapping[0:3]}")
        print(f"  O p-shells: {mapping[3:9]}")
        print(f"  O d-shell:  {mapping[9:14]}")
    else:
        print(f"  Partial mapping found ({len(mapping)}/24). Try finding per row sum?")
        # Maybe sorting by row-sum reveals groups?
        sum_p = np.sort(np.sum(np.abs(S_p), axis=1))
        sum_f = np.sort(np.sum(np.abs(S_f), axis=1))
        print(f"  Row sum match: {np.allclose(sum_p, sum_f, atol=1e-4)}")

check('GAUSSIAN')
check('QCHEM')
