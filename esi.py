import numpy as np

########### CORE ESIpy ###########

def aromaticity(mol, mf, Smo, rings, calc=None, mci=False, av1245=False, num_threads=None):
   
   symbols = [mol.atom_symbol(i) for i in range(mol.natm)]
   atom_numbers = [i + 1 for i in range(mol.natm)]

   ########### PRINTING THE OUTPUT ###########

   # Information from the calculation

   print(' -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+ ')
   print(' ** Localization & Delocalization Indices **  ')
   print(' **  For 3D Molecular Space Partitioning  **  ')
   print(' -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+ ')
   print('  Application to Aromaticity Calculations\n  Joan Grebol\n  See manual.pdf for citation of this program.')
   print(' -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+ ')
   print(' Number of Atoms:          {}'.format(mol.natm))
   if mf.__class__.__name__ == 'UHF' or mf.__class__.__name__ == 'UKS':
      print(' Occ. Mol. Orbitals:       {}({})'.format(int(mf.mo_occ[0].sum()), int(mf.mo_occ[1].sum())))
   else:
      print(' Occ. Mol. Orbitals:       {}({})'.format(int(mf.mo_occ.sum()), int(mf.mo_occ.sum())))
   print(' Wavefunction type:        hf')
   print(' Atomic partition:         Hilbert-space partition ({})'.format(calc.upper()))
   print(' ------------------------------------------- ')
   print(" Method:                  ", mf.__class__.__name__)
   if "dft" in mf.__module__ and mf.xc is not None:
      print(" Functional:              ", mf.xc)
   print(" Basis set:               ", mol.basis.upper())
   print(" Total energy:          {:>13f}".format(mf.e_tot))

   print(' ------------------------------------------- ')
   if mf.__class__.__name__ == 'UHF' or mf.__class__.__name__ == 'UKS':
      trace_alpha = np.sum([np.trace(matrix) for matrix in Smo[0]])
      trace_beta = np.sum([np.trace(matrix) for matrix in Smo[1]])
      print(' | Tr(alpha):    {:>13f}'.format(trace_alpha))
      print(' | Tr(beta):     {:>13f}'.format(trace_beta))
      print(' | Tr(total):    {:>13f}'.format(trace_alpha+trace_beta))
   else:
      trace = np.sum([np.trace(matrix) for matrix in Smo])
      print(' | Tr(Enter):    {:.13f}'.format(trace))
   print(' ------------------------------------------- ')
  

   # ATOMIC POPULATION ANALYSIS

   # UNRESTRICTED
   if mf.__class__.__name__ == 'UHF' or mf.__class__.__name__ == 'UKS':
   
      # Corresponding atom, total number of electrons from the AOMs (total, alpha and beta, respectively) and delocalization indices (alpha and beta, respectively)
      print(' ----------------------------------------------------------------------------- ')
      print(' |  Atom     N(Sij)     Na(Sij)     Nb(Sij)     Lapl.      dloc_a     dloc_b  ')
      print(' ----------------------------------------------------------------------------- ')

      Nij_alpha = []
      Nij_beta = []
      dloc_alpha = []
      dloc_beta = []
   
      for i in range(mol.natm):
         Nij_alpha.append(np.trace(Smo[0][i]))
         Nij_beta.append(np.trace(Smo[1][i]))
         dloc_alpha.append(np.trace(np.dot(Smo[0][i],Smo[0][i])))
         dloc_beta.append(np.trace(np.dot(Smo[1][i],Smo[1][i])))
         print(' | {} {:>2d}   {:10.6f}  {:10.6f}  {:10.6f}   *******   {:8.4f}   {:8.4f} '.format(mol.atom_symbol(i), i+1, np.trace(Smo[0][i])+np.trace(Smo[1][i]), np.trace(Smo[0][i]), np.trace(Smo[1][i]), np.trace(np.dot(Smo[0][i],Smo[0][i])), np.trace(np.dot(Smo[1][i],Smo[1][i]))))
      print(' ----------------------------------------------------------------------------- ')
      print(' | TOT:   {:10.6f}  {:10.6f}  {:10.6f}   *******   {:8.4f}   {:8.4f}'.format(sum(Nij_alpha)+sum(Nij_beta), sum(Nij_alpha), sum(Nij_beta), sum(dloc_alpha), sum(dloc_beta)))
      print(' ----------------------------------------------------------------------------- ')
      
   # RESTRICTED
   else:

      # Corresponding atom, total number of electrons from the AOMs, localization and delocalization indices for each atom
      print(' ------------------------------------------------------- ')
      print(' |  Atom    N(Sij)         Lapl.       loc.       dloc. ')
      print(' ------------------------------------------------------- ')

      Nij = []
      dloc = []

      for i in range(mol.natm):
         Nij.append(2*np.trace(Smo[i]))
         dloc.append(np.trace(np.dot(Smo[i],Smo[i])))
         print(' | {} {:>2d}    {:10.6f}     *******   {:8.4f}   {:8.4f} '.format(mol.atom_symbol(i), i+1, 2*np.trace(Smo[i]), 2*np.trace(np.dot(Smo[i],Smo[i])), 2*np.trace(Smo[i])-2*np.trace(np.dot(Smo[i],Smo[i]))))
      print(' ------------------------------------------------------- ')
      print(' | TOT:    {:10.6f}     *******   {:8.4f}   {:8.4f}'.format(sum(Nij), sum(Nij)-sum(dloc), sum(dloc)))
      print(' ------------------------------------------------------- ')
  

  # Writing the table with the DIs

   # UNRESTRICTED
   if mf.__class__.__name__ == 'UHF' or mf.__class__.__name__ == 'UKS':

      # Delocalization indices for each atom pair (total, alpha-alpha and beta-beta, respectively)
      print(' ------------------------------------------- ')
      print(' |    Pair         DI       DIaa      DIbb ')
      print(' ------------------------------------------- ')

      atom_charges = mol.atom_charges()
      dis_alpha = []
      dis_beta = []
      lis_alpha = []
      lis_beta = []

      for i in range(mol.natm):
         li_alpha = np.trace(np.dot(Smo[0][i], Smo[0][i]))
         li_beta = np.trace(np.dot(Smo[1][i], Smo[1][i]))
         lis_alpha.append(li_alpha)
         lis_beta.append(li_beta)

         for j in range(i+1, mol.natm):
            di_alpha = 2*np.trace(np.dot(Smo[0][i], Smo[0][j]))
            di_beta = 2*np.trace(np.dot(Smo[1][i], Smo[1][j]))
            dis_alpha.append(di_alpha)
            dis_beta.append(di_beta)
            print(' | {} {}-{} {}  {:>9.4f} {:>9.4f} {:>9.4f}'.format(mol.atom_symbol(i), str(i + 1).rjust(2), mol.atom_symbol(j), str(j + 1).rjust(2), di_alpha + di_beta, di_alpha, di_beta))
      print(' ------------------------------------------- ')
      print(' |    TOT:    {:>9.4f} {:>9.4f} {:>9.4f} '.format(sum(dis_alpha) + sum(dis_beta) + sum(lis_alpha) + sum(lis_beta), sum(dis_alpha) + sum(lis_alpha), sum(dis_beta) + sum(lis_beta)))
      print(' |    LOC:    {:>9.4f} {:>9.4f} {:>9.4f} '.format(sum(lis_alpha) + sum(lis_beta), sum(lis_alpha), sum(lis_beta)))
      print(' |  DELOC:    {:>9.4f} {:>9.4f} {:>9.4f} '.format(sum(dis_alpha) + sum(dis_beta), sum(dis_alpha), sum(dis_beta)))
   
   # RESTRICTED
   else:
      # Delocalization indices for each atom pair
      print(' ------------------------ ')
      print(' |    Pair         DI ')
      print(' ------------------------ ')

      atom_charges = mol.atom_charges()
      dis = []
      lis = []

      for i in range(mol.natm):
         li = 2*np.trace(np.dot(Smo[i], Smo[i]))
         lis.append(li)

         for j in range(i+1, mol.natm):
            di = 4*np.trace(np.dot(Smo[i], Smo[j]))
            dis.append(di)
            print(' | {} {}-{} {}   {:8.4f}'.format(mol.atom_symbol(i), str(i + 1).rjust(2), mol.atom_symbol(j), str(j + 1).rjust(2), di))
      print(' ------------------------ ')
      print(' |   TOT:      {:8.4f} '.format(np.sum(dis) + np.sum(lis)))
      print(' |   LOC:      {:8.4f} '.format(np.sum(lis)))
      print(' | DELOC:      {:8.4f} '.format(np.sum(dis)))
      print(' ------------------------ ')

   # Writing the aromaticity indicators

   print(" ----------------------------------------------------------------------")
   print(" | Aromaticity indices - PDI [CEJ 9, 400 (2003)]")
   print(" |                     Iring [PCCP 2, 3381 (2000)]")
   print(" |                    AV1245 [PCCP 18, 11839 (2016)]")
   print(" |                    AVmin  [JPCC 121, 27118 (2017)]")
   print(" |                           [PCCP 20, 2787 (2018)]")
   print(" |  For a recent review see: [CSR 44, 6434 (2015)]")
   print(" ----------------------------------------------------------------------")


   # Checking if the list rings is contains more than one ring to analyze

   if not isinstance(rings[0], list):
      rings = [rings]

   # Looping through each of the rings

   for ring_index, ring in enumerate(rings):

      print(" ----------------------------------------------------------------------")
      print(" |")
      print(" | Ring  {} ({}):   {}".format(ring_index + 1, len(ring), '  '.join(str(num) for num in ring)))
      print(" ----------------------------------------------------------------------")

      # Printing the PDI

      if len(ring) != 6:
         print(' |   PDI could not be calculated as the number of centers is not 6')

      else:

         # UNRESTRICTED
         if mf.__class__.__name__ == 'UHF' or mf.__class__.__name__ == 'UKS':
            pdis_alpha = compute_pdi(ring, Smo[0])
            pdis_beta = compute_pdi(ring, Smo[1])
            
            print(' | PDI_alpha  {} =         {:>9.4f} ( {:>9.4f} {:>9.4f} {:>9.4f} )'.format(ring_index + 1, pdis_alpha[0], pdis_alpha[1], pdis_alpha[2], pdis_alpha[3]))
            print(' | PDI_beta   {} =         {:>9.4f} ( {:>9.4f} {:>9.4f} {:>9.4f} )'.format(ring_index + 1, pdis_beta[0], pdis_beta[1], pdis_beta[2], pdis_beta[3]))
            print(' | PDI_total  {} =         {:>9.4f} ( {:>9.4f} {:>9.4f} {:>9.4f} )'.format(ring_index + 1, pdis_alpha[0]+pdis_beta[0], pdis_alpha[1]+pdis_beta[1], pdis_alpha[2]+pdis_beta[2], pdis_alpha[3]+pdis_beta[3]))
         
         # RESTRICTED
         else:   
            pdis = compute_pdi(ring, Smo)
            print(' | PDI  {} =             {:.4f} ( {:.4f} {:.4f} {:.4f} )'.format(ring_index + 1, pdis[0], pdis[1], pdis[2], pdis[3]))


      # Printing the AV1245 (if specified)

      if av1245 == True:
         print(' ---------------------------------------------------------------------- ')

         if len(ring) < 6:
            print(' | AV1245 could not be calculated as the number of centers is smaller than 6 ')

         else:

            # UNRESTRICTED
            if mf.__class__.__name__ == 'UHF' or mf.__class__.__name__ == 'UKS':
               avs_alpha = compute_av1245(ring, Smo[0])
               avs_beta = compute_av1245(ring, Smo[1])

               print(' |')
               print(' | *** AV1245_ALPHA ***')
               for j in range(len(ring)):
                  print(' |  {} {} - {} {} - {} {} - {} {}  |  {:>9.4f}'.format(
                     str(ring[j]).rjust(2), mol.atom_symbol((ring[j] - 1) % len(ring)),
                     str(ring[(j+1) % len(ring)]).rjust(2), mol.atom_symbol(ring[(j + 1) % len(ring)] - 1),
                     str(ring[(j+3) % len(ring)]).rjust(2), mol.atom_symbol(ring[(j + 3) % len(ring)] - 1),
                     str(ring[(j+4) % len(ring)]).rjust(2), mol.atom_symbol(ring[(j + 4) % len(ring)] - 1),
                     avs_alpha[2][(ring[j] - 1) % len(ring)]
                  ))
               print(' |   AV1245_alpha {} =             {:>9.4f}'.format(ring_index + 1, avs_alpha[0]))
               print(' |    AVmin_alpha {} =             {:>9.4f}'.format(ring_index + 1, avs_alpha[1]))

               print(' |')
               print(' | *** AV1245_BETA ***')

               for j in range(len(ring)):
                  print(' |  {} {} - {} {} - {} {} - {} {}  |  {:>9.4f}'.format(
                     str(ring[j]).rjust(2), mol.atom_symbol((ring[j] - 1) % len(ring)),
                     str(ring[(j+1) % len(ring)]).rjust(2), mol.atom_symbol(ring[(j + 1) % len(ring)] - 1),
                     str(ring[(j+3) % len(ring)]).rjust(2), mol.atom_symbol(ring[(j + 3) % len(ring)] - 1),
                     str(ring[(j+4) % len(ring)]).rjust(2), mol.atom_symbol(ring[(j + 4) % len(ring)] - 1),
                     avs_beta[2][(ring[j] - 1) % len(ring)]
                  ))
               print(' |   AV1245_beta  {} =             {:>9.4f}'.format(ring_index + 1, avs_beta[0]))
               print(' |    AVmin_beta  {} =             {:>9.4f}'.format(ring_index + 1, avs_beta[1]))
               print(' |')
               print(' | *** AV1245_TOTAL ***')
               print(' |   AV1245       {} =             {:>9.4f}'.format(ring_index + 1, avs_alpha[0]+avs_beta[0]))
               print(' |    AVmin       {} =             {:>9.4f}'.format(ring_index + 1, min(avs_alpha[1]+avs_beta[1], key=abs)))
               
            # RESTRICTED
            else:
               avs = 2 * compute_av1245(ring, Smo)

               for j in range(len(ring)):
                  print(' |  {} {} - {} {} - {} {} - {} {}  |  {:>6.4f}'.format(
                     str(ring[j]).rjust(2), mol.atom_symbol((ring[j] - 1) % len(ring)),
                     str(ring[(j+1) % len(ring)]).rjust(2), mol.atom_symbol(ring[(j + 1) % len(ring)] - 1),
                     str(ring[(j+3) % len(ring)]).rjust(2), mol.atom_symbol(ring[(j + 3) % len(ring)] - 1),
                     str(ring[(j+4) % len(ring)]).rjust(2), mol.atom_symbol(ring[(j + 4) % len(ring)] - 1),
                     avs[2][(ring[j] - 1) % len(ring)]
                  ))
               print(' | AV1245 {} =             {:.4f}'.format(ring_index + 1, avs[0]))
               print(' |  AVmin {} =             {:.4f}'.format(ring_index + 1, avs[1]))

      # Printing the Iring

      print(' ---------------------------------------------------------------------- ')

      # UNRESTRICTED
      if mf.__class__.__name__ == 'UHF' or mf.__class__.__name__ == 'UKS':
         iring_alpha = compute_iring(ring, Smo[0])
         iring_beta = compute_iring(ring, Smo[1])
         iring_total = iring_alpha + iring_beta

         print(' | Iring_alpha  {} =  {:>6f}'.format(ring_index + 1, iring_alpha))
         print(' | Iring_beta   {} =  {:>6f}'.format(ring_index + 1, iring_beta))
         print(' | Iring_total  {} =  {:>6f}'.format(ring_index + 1, iring_total))

         if iring_total < 0:
            print(' | Iring**(1/n) {} =  {:>6f}'.format(ring_index+1, -(np.abs(iring_total)**(1/len(ring)))))

         else:
            print(' | Iring**(1/n) {} =  {:>6f}'.format(ring_index+1, iring_total**(1/len(ring)))) 

      # RESTRICTED
      else:
         iring_total = compute_iring(ring, Smo)
         print(' | Iring        {} =  {:>.6f}'.format(ring_index + 1, iring_total))

         if iring < 0:
            print(' | Iring**(1/n) {} =  {:>.6f}'.format(ring_index+1, -(np.abs(iring_total)**(1/len(ring)))))

         else:
            print(' | Iring**(1/n) {} =  {:>.6f}'.format(ring_index+1, iring_total**(1/len(ring)))) 

      # Printing the MCI (if specified)

      if mci == True:
         import time
         print(' ---------------------------------------------------------------------- ')

         # SINGLE-CORE
         if num_threads == 1:

            # UNRESTRICTED
            if mf.__class__.__name__ == 'UHF' or mf.__class__.__name__ == 'UKS':

               start_mci = time.time()
               mci_alpha = sequential_mci(ring, Smo[0])
               mci_beta = sequential_mci(ring, Smo[1])
               mci_total = mci_alpha + mci_beta
               end_mci = time.time()
               time_mci = end_mci - start_mci
               print(" | The MCI calculation using 1 core took {:.4f} seconds".format(time_mci))
               print(' | MCI_alpha    {} =  {:>6f}'.format(ring_index + 1, mci_alpha))
               print(' | MCI_beta     {} =  {:>6f}'.format(ring_index + 1, mci_beta))
               print(' | MCI_total    {} =  {:>6f}'.format(ring_index + 1, mci_total))

            # RESTRICTED
            else:
               start_mci = time.time()
               mci_total = sequential_mci(ring, Smo)
               end_mci = time.time()
               time_mci = end_mci + start_mci

               print(" | The MCI calculation using 1 core took {:.4f} seconds".format(num_threads, time_mci))
               print(' | MCI          {} =  {:.6f}'.format(ring_index + 1, mci_total))
         
         # MULTI-CORE
         else:

            # UNRESTRICTED
            if mf.__class__.__name__ == 'UHF' or mf.__class__.__name__ == 'UKS':

               start_mci = time.time()
               mci_alpha = multiprocessing_mci(ring, Smo[0], num_threads)
               mci_beta = multiprocessing_mci(ring, Smo[1], num_threads)
               mci_total = mci_alpha + mci_beta
               end_mci = time.time()
               time_mci = end_mci - start_mci
               print(" | The MCI calculation using {} cores took {:.4f} seconds".format(num_threads, time_mci))
               print(' | MCI_alpha    {} =  {:>6f}'.format(ring_index + 1, mci_alpha))
               print(' | MCI_beta     {} =  {:>6f}'.format(ring_index + 1, mci_beta))
               print(' | MCI_total    {} =  {:>6f}'.format(ring_index + 1, mci_total))

            # RESTRICTED
            else:
               start_mci = time.time()
               mci_total = multiprocessing_mci(ring, Smo, num_threads)
               end_mci = time.time()
               time_mci = end_mci + start_mci

               print(" | The MCI calculation using {} cores took {:.4f} seconds".format(num_threads, time_mci))
               print(' | MCI          {} =  {:.6f}'.format(ring_index + 1, mci_total))
               
         if mci_total < 0:
            print(' | MCI**(1/n)   {} =  {:>6f}'.format(ring_index+1, -(np.abs(mci_total))**(1/len(ring))))

         else:
            print(' | MCI**(1/n)   {} =  {:>6f}'.format(ring_index+1, mci_total**(1/len(ring))))

      print(' ---------------------------------------------------------------------- ')


########### COMPUTATION OF THE AROMATICITY DESCRIPTORS ###########

########## Iring ###########

# Computing the Iring

def compute_iring(arr, Smo):

   product = np.identity(Smo[0].shape[0])

   for i in arr:
      product = np.dot(product, Smo[i-1])

   iring = 2**(len(arr)-1)*np.trace(product)

   return iring


########### MCI ###########

# Generating all the (n-1)!/2 non-cyclic permutations

def unique_permutations(nums):
   from itertools import permutations

   perm_set = set()

   for perm in permutations(nums[:-1]):
      if perm not in perm_set and perm[::-1] not in perm_set:
         perm_set.add(perm)

   perm_final = set()

   for perm in perm_set:
      perm_final.add(perm+(nums[-1],))

   return list(perm_final)


# MCI algorithm that does not store all the permutations (MCI2)

def sequential_mci(arr, Smo):

   mci_value = 0

   def generate_permutations(n, a, b, Smo):
      nonlocal mci_value
      iring = 0

      if n == 1:
         p=a + [b]
         product = np.linalg.multi_dot([Smo[i-1] for i in p])
         iring = 2**(len(arr)-2) * np.trace(product)
         mci_value += iring

      else:

         for i in range(n-1):
            generate_permutations(n-1, a, b, Smo)

            if n % 2 == 0:
               a[i], a[n-1] = a[n-1], a[i]

            else:
               a[0], a[n-1] = a[n-1], a[0]
         generate_permutations(n-1, a, b, Smo)

   generate_permutations(len(arr)-1, arr[:-1], arr[-1], Smo)

   return mci_value


# MCI algorithm that splits the job into different threads (MCI1)

def multiprocessing_mci(arr, Smo, num_threads):
   import multiprocessing as mp

   permutations = unique_permutations(arr)

   # split permutations into chunks for parallel processing
   chunk_size = int(np.ceil(len(permutations) / num_threads))
   permutations_chunks = [permutations[i:i+chunk_size] for i in range(0, len(permutations), chunk_size)]

   # create a pool of workers
   pool = mp.Pool(processes = num_threads)

   # compute the trace for each chunk in parallel
   results = []
   for chunk in permutations_chunks:
      result = pool.starmap(compute_iring, [(order, Smo) for order in chunk])
      results.extend(result)

   # sum up the results
   trace = sum(results)

   return trace

########### AV1245 ###########

# Calculation of the AV1245 index

def compute_av1245(arr, Smo):

   def av1245_pairs(arr):
      return [(arr[i % len(arr)], arr[(i+1) % len(arr)], arr[(i+3) % len(arr)], arr[(i+4) % len(arr)]) for i in range(len(arr))]

   min_product, av1245_value, avmin_value = 0, 0, 0
   val = 0
   product = 0
   products = []

   for cp in av1245_pairs(arr):
      product = sequential_mci(list(cp), Smo) 
      val += product
      products.append(1000 * product)

   min_product = min(products, key=abs)
   av1245_value = np.mean(products)
   avmin_value = min_product

   return [av1245_value, avmin_value, products]


########### PDI ###########

# Calculation of the PDI

def compute_pdi(arr, Smo):

   if len(arr)==6:
      pdi_a = 4*np.trace( np.dot( Smo[arr[0]-1], Smo[arr[3]-1]))
      pdi_b = 4*np.trace( np.dot( Smo[arr[1]-1], Smo[arr[4]-1]))
      pdi_c = 4*np.trace( np.dot( Smo[arr[2]-1], Smo[arr[5]-1]))
      pdi_value = (pdi_a + pdi_b + pdi_c)/3

      return [pdi_value, pdi_a, pdi_b, pdi_c]

   else:

      return None


########### COMPUTATION OF THE AROMATICITY DESCRIPTORS ###########

def aromaticity(mol, mf, Smo, rings, calc=None, mci=False, av1245=False, num_threads=None):
   
   symbols = [mol.atom_symbol(i) for i in range(mol.natm)]
   atom_numbers = [i + 1 for i in range(mol.natm)]

   ########### PRINTING THE OUTPUT ###########

   # Information from the calculation

   print(' -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+ ')
   print(' ** Localization & Delocalization Indices **  ')
   print(' **  For 3D Molecular Space Partitioning  **  ')
   print(' -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+ ')
   print('  Application to Aromaticity Calculations\n  Joan Grebol\n  See manual.pdf for citation of this program.')
   print(' -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+ ')
   print(' Number of Atoms:          {}'.format(mol.natm))
   if mf.__class__.__name__ == 'UHF' or mf.__class__.__name__ == 'UKS':
      print(' Occ. Mol. Orbitals:       {}({})'.format(int(mf.mo_occ[0].sum()), int(mf.mo_occ[1].sum())))
   else:
      print(' Occ. Mol. Orbitals:       {}({})'.format(int(mf.mo_occ.sum()), int(mf.mo_occ.sum())))
   print(' Wavefunction type:        hf')
   print(' Atomic partition:         Hilbert-space partition ({})'.format(calc.upper()))
   print(' ------------------------------------------- ')
   print(" Method:                  ", mf.__class__.__name__)
   if "dft" in mf.__module__ and mf.xc is not None:
      print(" Functional:              ", mf.xc)
   print(" Basis set:               ", mol.basis.upper())
   print(" Total energy:          {:>13f}".format(mf.e_tot))

   print(' ------------------------------------------- ')
   if mf.__class__.__name__ == 'UHF' or mf.__class__.__name__ == 'UKS':
      trace_alpha = np.sum([np.trace(matrix) for matrix in Smo[0]])
      trace_beta = np.sum([np.trace(matrix) for matrix in Smo[1]])
      print(' | Tr(alpha):    {:>13f}'.format(trace_alpha))
      print(' | Tr(beta):     {:>13f}'.format(trace_beta))
      print(' | Tr(total):    {:>13f}'.format(trace_alpha+trace_beta))
   else:
      trace = np.sum([np.trace(matrix) for matrix in Smo])
      print(' | Tr(Enter):    {:.13f}'.format(trace))
   print(' ------------------------------------------- ')
  

   # Writing the tables with the information related to the atomic populations analyses

   # UNRESTRICTED
   if mf.__class__.__name__ == 'UHF' or mf.__class__.__name__ == 'UKS':
   
      print(' ----------------------------------------------------------------------------- ')
      print(' |  Atom     N(Sij)     Na(Sij)     Nb(Sij)     Lapl.      dloc_a     dloc_b  ')
      print(' ----------------------------------------------------------------------------- ')

      Nij_alpha = []
      Nij_beta = []
      dloc_alpha = []
      dloc_beta = []
   
      for i in range(mol.natm):
         Nij_alpha.append(np.trace(Smo[0][i]))
         Nij_beta.append(np.trace(Smo[1][i]))
         dloc_alpha.append(np.trace(np.dot(Smo[0][i],Smo[0][i])))
         dloc_beta.append(np.trace(np.dot(Smo[1][i],Smo[1][i])))
         print(' | {} {:>2d}   {:10.6f}  {:10.6f}  {:10.6f}   *******   {:8.4f}   {:8.4f} '.format(mol.atom_symbol(i), i+1, np.trace(Smo[0][i])+np.trace(Smo[1][i]), np.trace(Smo[0][i]), np.trace(Smo[1][i]), np.trace(np.dot(Smo[0][i],Smo[0][i])), np.trace(np.dot(Smo[1][i],Smo[1][i]))))
      print(' ----------------------------------------------------------------------------- ')
      print(' | TOT:   {:10.6f}  {:10.6f}  {:10.6f}   *******   {:8.4f}   {:8.4f}'.format(sum(Nij_alpha)+sum(Nij_beta), sum(Nij_alpha), sum(Nij_beta), sum(dloc_alpha), sum(dloc_beta)))
      print(' ----------------------------------------------------------------------------- ')
      
   # RESTRICTED
   else:
      print(' ------------------------------------------------------- ')
      print(' |  Atom    N(Sij)         Lapl.       loc.       dloc. ')
      print(' ------------------------------------------------------- ')

      Nij = []
      dloc = []

      for i in range(mol.natm):
         Nij.append(2*np.trace(Smo[i]))
         dloc.append(np.trace(np.dot(Smo[i],Smo[i])))
         print(' | {} {:>2d}    {:10.6f}     *******   {:8.4f}   {:8.4f} '.format(mol.atom_symbol(i), i+1, 2*np.trace(Smo[i]), 2*np.trace(np.dot(Smo[i],Smo[i])), 2*np.trace(Smo[i])-2*np.trace(np.dot(Smo[i],Smo[i]))))
      print(' ------------------------------------------------------- ')
      print(' | TOT:    {:10.6f}     *******   {:8.4f}   {:8.4f}'.format(sum(Nij), sum(Nij)-sum(dloc), sum(dloc)))
      print(' ------------------------------------------------------- ')
  

  # Writing the table with the DIs

   # UNRESTRICTED
   if mf.__class__.__name__ == 'UHF' or mf.__class__.__name__ == 'UKS':

      print(' ------------------------------------------- ')
      print(' |    Pair         DI       DIaa      DIbb ')
      print(' ------------------------------------------- ')

      atom_charges = mol.atom_charges()
      dis_alpha = []
      dis_beta = []
      lis_alpha = []
      lis_beta = []

      for i in range(mol.natm):
         li_alpha = np.trace(np.dot(Smo[0][i], Smo[0][i]))
         li_beta = np.trace(np.dot(Smo[1][i], Smo[1][i]))
         lis_alpha.append(li_alpha)
         lis_beta.append(li_beta)

         for j in range(i+1, mol.natm):
            di_alpha = 2*np.trace(np.dot(Smo[0][i], Smo[0][j]))
            di_beta = 2*np.trace(np.dot(Smo[1][i], Smo[1][j]))
            dis_alpha.append(di_alpha)
            dis_beta.append(di_beta)
            print(' | {} {}-{} {}  {:>9.4f} {:>9.4f} {:>9.4f}'.format(mol.atom_symbol(i), str(i + 1).rjust(2), mol.atom_symbol(j), str(j + 1).rjust(2), di_alpha + di_beta, di_alpha, di_beta))
      print(' ------------------------------------------- ')
      print(' |    TOT:    {:>9.4f} {:>9.4f} {:>9.4f} '.format(sum(dis_alpha) + sum(dis_beta) + sum(lis_alpha) + sum(lis_beta), sum(dis_alpha) + sum(lis_alpha), sum(dis_beta) + sum(lis_beta)))
      print(' |    LOC:    {:>9.4f} {:>9.4f} {:>9.4f} '.format(sum(lis_alpha) + sum(lis_beta), sum(lis_alpha), sum(lis_beta)))
      print(' |  DELOC:    {:>9.4f} {:>9.4f} {:>9.4f} '.format(sum(dis_alpha) + sum(dis_beta), sum(dis_alpha), sum(dis_beta)))
   
   # RESTRICTED
   else:
      print(' ------------------------ ')
      print(' |    Pair         DI ')
      print(' ------------------------ ')

      atom_charges = mol.atom_charges()
      dis = []
      lis = []

      for i in range(mol.natm):
         li = 2*np.trace(np.dot(Smo[i], Smo[i]))
         lis.append(li)

         for j in range(i+1, mol.natm):
            di = 4*np.trace(np.dot(Smo[i], Smo[j]))
            dis.append(di)
            print(' | {} {}-{} {}   {:8.4f}'.format(mol.atom_symbol(i), str(i + 1).rjust(2), mol.atom_symbol(j), str(j + 1).rjust(2), di))
      print(' ------------------------ ')
      print(' |   TOT:      {:8.4f} '.format(np.sum(dis) + np.sum(lis)))
      print(' |   LOC:      {:8.4f} '.format(np.sum(lis)))
      print(' | DELOC:      {:8.4f} '.format(np.sum(dis)))
      print(' ------------------------ ')

   # Writing the aromaticity indicators

   print(" ----------------------------------------------------------------------")
   print(" | Aromaticity indices - PDI [CEJ 9, 400 (2003)]")
   print(" |                     Iring [PCCP 2, 3381 (2000)]")
   print(" |                    AV1245 [PCCP 18, 11839 (2016)]")
   print(" |                    AVmin  [JPCC 121, 27118 (2017)]")
   print(" |                           [PCCP 20, 2787 (2018)]")
   print(" |  For a recent review see: [CSR 44, 6434 (2015)]")
   print(" ----------------------------------------------------------------------")


   # Checking if the list rings is contains more than one ring to analyze

   if not isinstance(rings[0], list):
      rings = [rings]

   # Looping through each of the rings

   for ring_index, ring in enumerate(rings):

      print(" ----------------------------------------------------------------------")
      print(" |")
      print(" | Ring  {} ({}):   {}".format(ring_index + 1, len(ring), '  '.join(str(num) for num in ring)))
      print(" ----------------------------------------------------------------------")

      # Printing the PDI

      if len(ring) != 6:
         print(' |   PDI could not be calculated as the number of centers is not 6')

      else:

         # UNRESTRICTED
         if mf.__class__.__name__ == 'UHF' or mf.__class__.__name__ == 'UKS':
            pdis_alpha = compute_pdi(ring, Smo[0])
            pdis_beta = compute_pdi(ring, Smo[1])
            
            print(' | PDI_alpha  {} =         {:>9.4f} ( {:>9.4f} {:>9.4f} {:>9.4f} )'.format(ring_index + 1, pdis_alpha[0], pdis_alpha[1], pdis_alpha[2], pdis_alpha[3]))
            print(' | PDI_beta   {} =         {:>9.4f} ( {:>9.4f} {:>9.4f} {:>9.4f} )'.format(ring_index + 1, pdis_beta[0], pdis_beta[1], pdis_beta[2], pdis_beta[3]))
            print(' | PDI_total  {} =         {:>9.4f} ( {:>9.4f} {:>9.4f} {:>9.4f} )'.format(ring_index + 1, pdis_alpha[0]+pdis_beta[0], pdis_alpha[1]+pdis_beta[1], pdis_alpha[2]+pdis_beta[2], pdis_alpha[3]+pdis_beta[3]))
         
         # RESTRICTED
         else:   
            pdis = compute_pdi(ring, Smo)
            print(' | PDI  {} =             {:.4f} ( {:.4f} {:.4f} {:.4f} )'.format(ring_index + 1, pdis[0], pdis[1], pdis[2], pdis[3]))


      # Printing the AV1245 (if specified)

      if av1245 == True:
         print(' ---------------------------------------------------------------------- ')

         if len(ring) < 6:
            print(' | AV1245 could not be calculated as the number of centers is smaller than 6 ')

         else:

            # UNRESTRICTED
            if mf.__class__.__name__ == 'UHF' or mf.__class__.__name__ == 'UKS':
               avs_alpha = compute_av1245(ring, Smo[0])
               avs_beta = compute_av1245(ring, Smo[1])

               print(' |')
               print(' | *** AV1245_ALPHA ***')
               for j in range(len(ring)):
                  print(' |  {} {} - {} {} - {} {} - {} {}  |  {:>9.4f}'.format(
                     str(ring[j]).rjust(2), mol.atom_symbol((ring[j] - 1) % len(ring)),
                     str(ring[(j+1) % len(ring)]).rjust(2), mol.atom_symbol(ring[(j + 1) % len(ring)] - 1),
                     str(ring[(j+3) % len(ring)]).rjust(2), mol.atom_symbol(ring[(j + 3) % len(ring)] - 1),
                     str(ring[(j+4) % len(ring)]).rjust(2), mol.atom_symbol(ring[(j + 4) % len(ring)] - 1),
                     avs_alpha[2][(ring[j] - 1) % len(ring)]
                  ))
               print(' |   AV1245_alpha {} =             {:>9.4f}'.format(ring_index + 1, avs_alpha[0]))
               print(' |    AVmin_alpha {} =             {:>9.4f}'.format(ring_index + 1, avs_alpha[1]))

               print(' |')
               print(' | *** AV1245_BETA ***')

               for j in range(len(ring)):
                  print(' |  {} {} - {} {} - {} {} - {} {}  |  {:>9.4f}'.format(
                     str(ring[j]).rjust(2), mol.atom_symbol((ring[j] - 1) % len(ring)),
                     str(ring[(j+1) % len(ring)]).rjust(2), mol.atom_symbol(ring[(j + 1) % len(ring)] - 1),
                     str(ring[(j+3) % len(ring)]).rjust(2), mol.atom_symbol(ring[(j + 3) % len(ring)] - 1),
                     str(ring[(j+4) % len(ring)]).rjust(2), mol.atom_symbol(ring[(j + 4) % len(ring)] - 1),
                     avs_beta[2][(ring[j] - 1) % len(ring)]
                  ))
               print(' |   AV1245_beta  {} =             {:>9.4f}'.format(ring_index + 1, avs_beta[0]))
               print(' |    AVmin_beta  {} =             {:>9.4f}'.format(ring_index + 1, avs_beta[1]))
               print(' |')
               print(' | *** AV1245_TOTAL ***')
               print(' |   AV1245       {} =             {:>9.4f}'.format(ring_index + 1, avs_alpha[0]+avs_beta[0]))
               print(' |    AVmin       {} =             {:>9.4f}'.format(ring_index + 1, min(avs_alpha[1]+avs_beta[1], key=abs)))
               
            # RESTRICTED
            else:
               avs = 2 * compute_av1245(ring, Smo)

               for j in range(len(ring)):
                  print(' |  {} {} - {} {} - {} {} - {} {}  |  {:>6.4f}'.format(
                     str(ring[j]).rjust(2), mol.atom_symbol((ring[j] - 1) % len(ring)),
                     str(ring[(j+1) % len(ring)]).rjust(2), mol.atom_symbol(ring[(j + 1) % len(ring)] - 1),
                     str(ring[(j+3) % len(ring)]).rjust(2), mol.atom_symbol(ring[(j + 3) % len(ring)] - 1),
                     str(ring[(j+4) % len(ring)]).rjust(2), mol.atom_symbol(ring[(j + 4) % len(ring)] - 1),
                     avs[2][(ring[j] - 1) % len(ring)]
                  ))
               print(' | AV1245 {} =             {:.4f}'.format(ring_index + 1, avs[0]))
               print(' |  AVmin {} =             {:.4f}'.format(ring_index + 1, avs[1]))

      # Printing the Iring

      print(' ---------------------------------------------------------------------- ')

      # UNRESTRICTED
      if mf.__class__.__name__ == 'UHF' or mf.__class__.__name__ == 'UKS':
         iring_alpha = compute_iring(ring, Smo[0])
         iring_beta = compute_iring(ring, Smo[1])
         iring_total = iring_alpha + iring_beta

         print(' | Iring_alpha  {} =  {:>6f}'.format(ring_index + 1, iring_alpha))
         print(' | Iring_beta   {} =  {:>6f}'.format(ring_index + 1, iring_beta))
         print(' | Iring_total  {} =  {:>6f}'.format(ring_index + 1, iring_total))

         if iring_total < 0:
            print(' | Iring**(1/n) {} =  {:>6f}'.format(ring_index+1, -(np.abs(iring_total)**(1/len(ring)))))

         else:
            print(' | Iring**(1/n) {} =  {:>6f}'.format(ring_index+1, iring_total**(1/len(ring)))) 

      # RESTRICTED
      else:
         iring_total = compute_iring(ring, Smo)
         print(' | Iring        {} =  {:>.6f}'.format(ring_index + 1, iring_total))

         if iring < 0:
            print(' | Iring**(1/n) {} =  {:>.6f}'.format(ring_index+1, -(np.abs(iring_total)**(1/len(ring)))))

         else:
            print(' | Iring**(1/n) {} =  {:>.6f}'.format(ring_index+1, iring_total**(1/len(ring)))) 

      # Printing the MCI (if specified)

      if mci == True:
         import time
         print(' ---------------------------------------------------------------------- ')

         # SINGLE-CORE
         if num_threads == 1:

            # UNRESTRICTED
            if mf.__class__.__name__ == 'UHF' or mf.__class__.__name__ == 'UKS':

               start_mci = time.time()
               mci_alpha = sequential_mci(ring, Smo[0])
               mci_beta = sequential_mci(ring, Smo[1])
               mci_total = mci_alpha + mci_beta
               end_mci = time.time()
               time_mci = end_mci - start_mci
               print(" | The MCI calculation using 1 core took {:.4f} seconds".format(time_mci))
               print(' | MCI_alpha    {} =  {:>6f}'.format(ring_index + 1, mci_alpha))
               print(' | MCI_beta     {} =  {:>6f}'.format(ring_index + 1, mci_beta))
               print(' | MCI_total    {} =  {:>6f}'.format(ring_index + 1, mci_total))

            # RESTRICTED
            else:
               start_mci = time.time()
               mci_total = sequential_mci(ring, Smo)
               end_mci = time.time()
               time_mci = end_mci + start_mci

               print(" | The MCI calculation using 1 core took {:.4f} seconds".format(num_threads, time_mci))
               print(' | MCI          {} =  {:.6f}'.format(ring_index + 1, mci_total))
         
         # MULTI-CORE
         else:

            # UNRESTRICTED
            if mf.__class__.__name__ == 'UHF' or mf.__class__.__name__ == 'UKS':

               start_mci = time.time()
               mci_alpha = multiprocessing_mci(ring, Smo[0], num_threads)
               mci_beta = multiprocessing_mci(ring, Smo[1], num_threads)
               mci_total = mci_alpha + mci_beta
               end_mci = time.time()
               time_mci = end_mci - start_mci
               print(" | The MCI calculation using {} cores took {:.4f} seconds".format(num_threads, time_mci))
               print(' | MCI_alpha    {} =  {:>6f}'.format(ring_index + 1, mci_alpha))
               print(' | MCI_beta     {} =  {:>6f}'.format(ring_index + 1, mci_beta))
               print(' | MCI_total    {} =  {:>6f}'.format(ring_index + 1, mci_total))

            # RESTRICTED
            else:
               start_mci = time.time()
               mci_total = multiprocessing_mci(ring, Smo, num_threads)
               end_mci = time.time()
               time_mci = end_mci + start_mci

               print(" | The MCI calculation using {} cores took {:.4f} seconds".format(num_threads, time_mci))
               print(' | MCI          {} =  {:.6f}'.format(ring_index + 1, mci_total))
               
         if mci_total < 0:
            print(' | MCI**(1/n)   {} =  {:>6f}'.format(ring_index+1, -(np.abs(mci_total))**(1/len(ring))))

         else:
            print(' | MCI**(1/n)   {} =  {:>6f}'.format(ring_index+1, mci_total**(1/len(ring))))

      print(' ---------------------------------------------------------------------- ')



def make_aom(mol,mf,calc=None):
   from pyscf import lo
   import numpy as np
   import os
   
   if mf.__class__.__name__ == 'UHF' or mf.__class__.__name__ == 'UKS':
   
      # Getting specific information

      natom = mol.natm
      nbas = mol.nao
      S = mf.get_ovlp()
      nocc_alpha = mf.mo_occ[0].astype(int)
      nocc_beta = mf.mo_occ[1].astype(int)
      occ_coeff_alpha = mf.mo_coeff[0][:, :nocc_alpha.sum()]
      occ_coeff_beta = mf.mo_coeff[0][:, :nocc_beta.sum()]

      # Building the Atomic Overlap Matrices

      Smo_alpha = []
      Smo_beta = []

      if calc == 'lowdin' or calc == 'meta_lowdin' or calc == 'nao':

         U_inv = lo.orth_ao(mf, calc, pre_orth_ao = None)
         U = np.linalg.inv(U_inv)

         eta = [np.zeros((mol.nao, mol.nao)) for i in range(mol.natm)]
         for i in range(mol.natm):
            start = mol.aoslice_by_atom()[i, -2]
            end = mol.aoslice_by_atom()[i, -1]
            eta[i][start:end, start:end] = np.eye(end-start)

         for i in range(mol.natm):
            SCR_alpha = np.linalg.multi_dot((occ_coeff_alpha.T, U.T, eta[i]))
            SCR_beta = np.linalg.multi_dot((occ_coeff_beta.T, U.T, eta[i]))
            Smo_alpha.append(np.dot(SCR_alpha,SCR_alpha.T))
            Smo_beta.append(np.dot(SCR_beta,SCR_beta.T))

      # Special case IAO
      elif calc == 'iao':

         U_alpha_iao_nonortho = lo.iao.iao(mol, occ_coeff_alpha)
         U_beta_iao_nonortho = lo.iao.iao(mol, occ_coeff_beta)
         U_alpha_inv=np.dot(U_alpha_iao_nonortho, lo.orth.lowdin(np.linalg.multi_dot((U_alpha_iao_nonortho.T, S, U_alpha_iao_nonortho))))
         U_beta_inv=np.dot(U_beta_iao_nonortho, lo.orth.lowdin(np.linalg.multi_dot((U_beta_iao_nonortho.T, S, U_ibeta_iao_nonortho))))
         U_alpha = np.dot(S, U_alpha_inv)
         U_beta = np.dot(S, U_beta_inv)
         pmol = lo.iao.reference_mol(mol)
         nbas_iao = pmol.nao

         eta = [np.zeros((pmol.nao, pmol.nao)) for i in range(pmol.natm)]
         for i in range(pmol.natm):
            start = pmol.aoslice_by_atom()[i, -2]
            end = pmol.aoslice_by_atom()[i, -1]
            eta[i][start:end, start:end] = np.eye(end-start)

         for i in range(pmol.natm):
            SCR_alpha = np.linalg.multi_dot((occ_coeff_alpha.T, U_alpha, eta[i]))
            SCR_beta = np.linalg.multi_dot((occ_coeff_beta.T, U_beta, eta[i]))
            Smo_alpha.append(np.dot(SCR_alpha, SCR_alpha.T))
            Smo_beta.append(np.dot(SCR_beta, SCR_beta.T))

      # Special case plain Mulliken
      elif calc == 'mulliken':

         eta = [np.zeros((mol.nao, mol.nao)) for i in range(mol.natm)]
         for i in range(mol.natm):
            start = mol.aoslice_by_atom()[i, -2]
            end = mol.aoslice_by_atom()[i, -1]
            eta[i][start:end, start:end] = np.eye(end-start)

         for i in range(mol.natm):
            SCR_alpha = np.linalg.multi_dot((occ_coeff_alpha.T, S, eta[i], occ_coeff_alpha))
            SCR_beta = np.linalg.multi_dot((occ_coeff_beta.T, S, eta[i], occ_coeff_beta))
            Smo_alpha.append(SCR_alpha)
            Smo_beta.append(SCR_beta)

      else:
         raise NameError('Hilbert-space scheme not available')

      print('shape Smo alhpa', np.shape(Smo_alpha))
      return [Smo_alpha, Smo_beta]

   else:

      # Getting specific information

      natom = mol.natm
      nbas = mol.nao
      S = mf.get_ovlp()
      occ_coeff = mf.mo_coeff[:, mf.mo_occ > 0] 

      # Building the Atomic Overlap Matrices

      Smo = []

      if calc == 'lowdin' or calc == 'meta_lowdin' or calc == 'nao':

         U_inv = lo.orth_ao(mf, calc, pre_orth_ao = None)
         U = np.linalg.inv(U_inv)

         eta = [np.zeros((mol.nao, mol.nao)) for i in range(mol.natm)]
         for i in range(mol.natm):
            start = mol.aoslice_by_atom()[i, -2]
            end = mol.aoslice_by_atom()[i, -1]
            eta[i][start:end, start:end] = np.eye(end-start)

         for i in range(mol.natm):
            SCR = np.linalg.multi_dot((occ_coeff.T, U.T, eta[i]))
            Smo.append(np.dot(SCR,SCR.T))

      # Special case IAO
      elif calc == 'iao':

         U_iao_nonortho = lo.iao.iao(mol, occ_coeff)
         U_inv=np.dot(U_iao_nonortho, lo.orth.lowdin(np.linalg.multi_dot((U_iao_nonortho.T, S, U_iao_nonortho))))
         U = np.dot(S, U_inv)
         pmol = lo.iao.reference_mol(mol)
         nbas_iao = pmol.nao

         eta = [np.zeros((pmol.nao, pmol.nao)) for i in range(pmol.natm)]
         for i in range(pmol.natm):
            start = pmol.aoslice_by_atom()[i, -2]
            end = pmol.aoslice_by_atom()[i, -1]
            eta[i][start:end, start:end] = np.eye(end-start)

         for i in range(pmol.natm):
            SCR = np.linalg.multi_dot((occ_coeff.T, U, eta[i]))
            Smo.append(np.dot(SCR, SCR.T))

      # Special case plain Mulliken
      elif calc == 'mulliken':

         eta = [np.zeros((mol.nao, mol.nao)) for i in range(mol.natm)]
         for i in range(mol.natm):
            start = mol.aoslice_by_atom()[i, -2]
            end = mol.aoslice_by_atom()[i, -1]
            eta[i][start:end, start:end] = np.eye(end-start)

         for i in range(mol.natm):
            SCR = np.linalg.multi_dot((occ_coeff.T, S, eta[i], occ_coeff))
            Smo.append(SCR)

      else:
         raise NameError('Hilbert-space scheme not available')

      return Smo 


########### WRITING THE INPUT FOR THE ESI-3D CODE FROM THE AOMS ###########

def write_int(mol, mf, molname, Smo, ring=None, calc=None):
   import os
  
   # Obtaining information for the files

   symbols = [mol.atom_symbol(i) for i in range(mol.natm)]
   atom_numbers = [i + 1 for i in range(mol.natm)]
   charge = mol.atom_charges()
   if mf.__class__.__name__ == 'UHF' or mf.__class__.__name__ == 'UKS':
      nocc_alpha = mf.mo_occ[0].astype(int)
      nocc_beta = mf.mo_occ[1].astype(int)
      occ_coeff_alpha = mf.mo_coeff[0][:, :nocc_alpha.sum()]
      occ_coeff_beta = mf.mo_coeff[0][:, :nocc_beta.sum()]
      nalpha = [(charge + np.trace(aom_alpha)) / 2 for aom_alpha in Smo[0]]
      nbeta = [(charge + np.trace(aom_beta)) / 2 for aom_beta in Smo[1]]

      # Creating the zero-matrix to fill the print of the AOM
      Smos = []
      fill = np.zeros((nocc_beta.sum(), nocc_alpha.sum()))
      for i in range(mol.natm):
         left = np.vstack((Smo[0][i],fill))
         right = np.vstack((fill.T,Smo[1][i]))
         matrix = np.hstack((left,right))
         Smos.append(matrix)

   else:
      nalpha = nbeta = [(charge + np.trace(aom)) / 2 for aom in Smo]

   # Creating a new directory for the calculation

   if calc == 'mulliken':
      shortcalc = 'mul'
   elif calc == 'lowdin':
      shortcalc = 'low'
   elif calc == 'meta_lowdin':
      shortcalc = 'metalow'
   elif calc == 'nao':
      shortcalc = 'nao'
   elif calc == 'iao':
      shortcalc = 'iao'
   else:
     raise NameError('Hilbert-space scheme not available')

   new_dir_name = molname + shortcalc
   titles = [symbols[i] + str(atom_numbers[i]) + shortcalc for i in range(mol.natm)] #Setting the title of the files
   new_dir_path = os.path.join(os.getcwd(), new_dir_name)
   os.makedirs(new_dir_path, exist_ok=True)

   # Creating and writing the atomic .int files

   for i, item in enumerate(titles):
       with open(os.path.join(new_dir_path, item + '.int'), 'w+') as f:
         f.write(' Created by ESIpy\n')
         if calc=='mulliken':
            f.write(' Using Mulliken atomic definition\n')
         elif calc=='lowdin':
            f.write(' Using Lowdin atomic definition\n')
         elif calc=='meta_lowdin':
            f.write(' Using Meta-Lowdin atomic definition\n')
         elif calc=='nao':
            f.write(' Using NAO atomic definition\n')
         elif calc=='iao':
            f.write(' Using IAO atomic definition\n')
         f.write(' Single-determinant wave function\n')
         f.write(' Molecular SCF ENERGY (AU)  =       {:.11f}\n\n'.format(mf.energy_tot()))
         f.write(' INTEGRATION IS OVER ATOM  {}    {}\n'.format(symbols[i], i+1))
         f.write(' RESULTS OF THE INTEGRATION\n')
         if mf.__class__.__name__ == 'UHF' or mf.__class__.__name__ == 'UKS':
            f.write('              N   {:.14E}    NET CHARGE {:.14E}\n'.format(1,1))
         else:
            f.write('              N   {:.14E}    NET CHARGE {:.14E}\n'.format(2*np.trace(Smo[i]),round(charge[i]-2*np.trace(Smo[i]),14)))
         f.write('              G\n')
         f.write('              K   1.00000000000000E+01        E(ATOM)  1.00000000000000E+00\n')
         f.write('              L   1.00000000000000E+01\n\n')

         if mf.__class__.__name__ == 'UHF' or mf.__class__.__name__ == 'UKS':
            f.write('\n The Atomic Overlap Matrix:\n\nUnrestricted\n\n')
            f.write('\n'.join(['  '.join(['{:.16E}'.format(Smos[i][j][k]) if j >= k else '' for k in range(len(Smos[i][j]))]) for j in range(len(Smos[i]))])+'\n')

         else:
            f.write('\n          The Atomic Overlap Matrix\n\nRestricted Closed-Shell Wavefunction\n\n  ')
            if calc=='mulliken':
               f.write('  \n'.join(['  '.join(['{:.16E}'.format(num,16) for num in row]) for row in Smo[i]])+'\n')
            else:
               f.write('\n'.join(['  '.join(['{:.16E}'.format(Smo[i][j][k],16) if j >= k else '' for k in range(len(Smo[i][j]))]) for j in range(len(Smo[i]))])+'\n')
         f.write('\n                     ALPHA ELECTRONS (NA) {:E}\n'.format(nalpha[i][0], 14))
         f.write('                      BETA ELECTRONS (NB) {:E}\n\n'.format(nbeta[i][0], 14))
         f.write(' NORMAL TERMINATION OF PROAIMV')
         f.close()

   # Writing the file containing the title of the atomic .int files

   with open(os.path.join(new_dir_path, molname + shortcalc + '.files'), 'w') as f:
      for i in titles:
         f.write(i + '.int\n')
      f.close()
   # Creating the input for the ESI-3D code

   filename = os.path.join(new_dir_path, molname + ".bad")
   with open(filename, "w") as f:
      f.write("$TITLE\n")
      f.write(molname + "\n")
      f.write("$TYPE\n")
      if mf.__class__.__name__ == 'UHF' or mf.__class__.__name__ == 'UKS':
         f.write("uhf\n{}\n".format(mol.nelec[0]+1))
      else:
         f.write("hf\n")
      #f.write("$NOMCI\n")
      #f.write("$MCIALG\n1\n")
      f.write("$RING\n")
      if ring is not None:
         if isinstance(ring[0],int): # If only one ring is specified
            f.write("1\n{}\n".format(len(ring)))
            f.write(" ".join(str(value) for value in ring))
            f.write("\n")
         else:
            f.write("{}\n".format(len(ring))) # If two or more rings are specified as a list of lists
            for sublist in ring:
               f.write(str(len(sublist)) + "\n")
               f.write(" ".join(str(value) for value in sublist))
               f.write("\n")
      else:
         f.write("\n") # No ring specified, write it manually
      f.write("$ATOMS\n")
      f.write(str(mol.natm) + "\n")
      for title in titles:
         f.write(title + ".int\n")
      f.write("$BASIS\n")
      f.write(str(int(mf.mo_occ[0].sum())+int(mf.mo_occ[1].sum())) + "\n")
      f.write("$AV1245\n")
      f.write("$FULLOUT\n")
      if calc == 'mulliken':
         f.write("$MULLIKEN\n")








