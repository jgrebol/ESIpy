import esi
from utils import ints

Smo = ints.read_int('./example08_nao/')
print(Smo)
print(len(Smo))
print(len(Smo[0]))
    
esi.aromaticity(Smo, rings=[7,3,1,2,6,10], mci=True, av1245=True)
