import esi
from utils import ints

Smo = ints.read_aoms('aimall', './example08_nao/')

esi.aromaticity(Smo, rings=[7,3,1,2,6,10], mci=True, av1245=True)
