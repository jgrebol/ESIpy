import esipy

# As an exercise, we use the HOMA references into the HOMER calculation
# The HOMER value is the same, therefore, as the HOMA one
# For custom homarefs, user should provide references as in:
# J. Kruszewski and T. M. Krygowski. Tetrahedron Lett., 13(36):3839–3842, 1972

homerref = {'CC': {'r_opt': 1.388, 'alpha': 257.7}}
connectivity = ['C', 'C', 'C', 'C', 'C', 'C']
ring = [1, 2, 3, 4, 5, 6]
partition = 'nao'

# The geometry can be directly extracted from the mol.atom_coords() method
geom = [[-4.25275075,  0.00000000, -1.81149504],
        [-4.25275075,  2.28270034, -3.25176304],
        [-4.25275075,  2.28270034, -5.95240466],
        [-4.25275075,  0.00000000, -7.39267265],
        [-4.25275075, -2.28270034, -5.95240466],
        [-4.25275075, -2.28270034, -3.25176304],
        [-4.25275075,  4.10765610, -2.28997181],
        [-4.25275075, -4.10765610, -2.28997181],
        [-4.25275075,  4.10765610, -6.91419589],
        [-4.25275075, -4.10765610, -6.91419589],
        [-4.25275075,  0.00000000, -9.44284219],
        [-4.25275075,  0.00000000,  0.23867450]]
molinfo = 'example01_nao.molinfo'
aom = 'example01_nao.aoms'

esipy.ESI(aom=aom, molinfo=molinfo, rings=ring, partition=partition, homerrefs=homerref, connectivity=connectivity,
          geom=geom).print()
