import re

def fix():
    with open('tools.py', 'r') as f:
        content = f.read()

    # Restore the EINSUM_BACKEND hack for post-HF
    hack_code = """    if 'ccsd' in str(mf.__class__).lower() or 'mp2' in str(mf.__class__).lower():
        from pyscf import lib
        old_backend = getattr(lib.numpy_helper, 'EINSUM_BACKEND', 'pyscf')
        lib.numpy_helper.EINSUM_BACKEND = 'numpy'
        try:
            rdm1 = mf.make_rdm1(ao_repr=True)
        finally:
            lib.numpy_helper.EINSUM_BACKEND = old_backend
    else:
        rdm1 = mf.make_rdm1(ao_repr=True)"""

    # Currently it has:
    #     rdm1 = mf.make_rdm1(ao_repr=True)
    # right after print(" | Obtaining Natural Orbitals from the 1-RDM...")
    
    pattern = re.compile(r"    rdm1 = mf\.make_rdm1\(ao_repr=True\)\n    rdm1_arr = np\.asarray\(rdm1\)")
    content = pattern.sub(hack_code + "\n    rdm1_arr = np.asarray(rdm1)", content)

    with open('tools.py', 'w') as f:
        f.write(content)

if __name__ == "__main__":
    fix()
