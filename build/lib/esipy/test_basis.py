from pyscf import gto
import sys

try:
    basis = gto.basis.load('ano', 'C')
    print(basis)
except Exception as e:
    print(f"Error loading 'ano': {e}")
    try:
        basis = gto.basis.load('ano-rcc', 'C')
        print("Loaded 'ano-rcc' instead.")
    except Exception as e2:
        print(f"Error loading 'ano-rcc': {e2}")
