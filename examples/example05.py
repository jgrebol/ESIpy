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
geom = [[-2.58138047, -1.26965218, -5.69564833],
        [-4.98179941, -2.68686741, -5.69564833],
        [-7.38220701, -1.26965218, -5.69564833],
        [-7.38220701, 1.26965218, -5.69564833],
        [-4.98178807, 2.68686741, -5.69564833],
        [-2.58138047, 1.26965218, -5.69564833],
        [-0.80908759, -2.32169862, -5.69564833],
        [-4.98179941, -4.74141922, -5.69564833],
        [-9.15450366, -2.32169484, -5.69564833],
        [-9.15449988, 2.32169862, -5.69564833],
        [-4.98178807, 4.74141922, -5.69564833],
        [-0.80908381, 2.32169484, -5.69564833]]
molinfo = 'example01_nao.molinfo'
Smo = 'example01_nao.aoms'

esipy.ESI(Smo=Smo, molinfo=molinfo, rings=ring, partition=partition, homerrefs=homerref, connectivity=connectivity,
          geom=geom).print()
