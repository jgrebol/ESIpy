import time
from pyscf import gto, scf, fci, mcscf

# Same basis as user provided (aug-cc-pVDZ)
basis_lih = {
    'H': [
        [0, [13.01, 0.019685], [1.962, 0.137977], [0.4446, 0.478148], [0.122, 0.50124]],
        [0, [0.122, 1.0]],
        [0, [0.02974, 1.0]],
        [1, [0.727, 1.0]],
        [1, [0.141, 1.0]]
    ],
    'Li': [
        [0, [1469.0, 0.000766], [220.5, 0.005892], [50.26, 0.029671], [14.24, 0.10918], [4.581, 0.282789], [1.58, 0.453123], [0.564, 0.274774], [0.07345, 0.009751], [0.02805, -0.00318]],
        [0, [1469.0, -0.00012], [220.5, -0.000923], [50.26, -0.004689], [14.24, -0.017682], [4.581, -0.048902], [1.58, -0.096009], [0.564, -0.13638], [0.07345, 0.575102], [0.02805, 0.517661]],
        [0, [0.02805, 1.0]],
        [0, [0.0086, 1.0]],
        [1, [1.534, 0.022784], [0.2749, 0.139107], [0.07362, 0.500375], [0.02403, 0.508474]],
        [1, [0.02403, 1.0]],
        [1, [0.0058, 1.0]],
        [2, [0.1144, 1.0]],
        [2, [0.0733, 1.0]]
    ]
}

mol = gto.M(atom='Li 0 0 0; H 0 0 1.6', basis=basis_lih, verbose=0)
print(f"Number of basis functions: {mol.nao_nr()}")

# 1. HF
start = time.time()
mf = scf.RHF(mol).run()
print(f"HF Time: {time.time() - start:.4f} s")

# 2. FCI
start = time.time()
cisolver = fci.FCI(mf)
e_fci, fcivec = cisolver.kernel()
print(f"FCI Time: {time.time() - start:.4f} s")
print(f"FCI Energy: {e_fci:.8f}")

# 3. CASCI equivalent (Full Active Space)
start = time.time()
mycas = mcscf.CASCI(mf, mol.nao_nr(), 4)
e_casci = mycas.kernel()[0]
print(f"CASCI(4,{mol.nao_nr()}) Time: {time.time() - start:.4f} s")
print(f"CASCI Energy: {e_casci:.8f}")

