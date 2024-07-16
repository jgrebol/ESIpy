import os
import time
import re
import numpy as np

def read_aoms(source, path='.'):
   if source == 'esipy':
      Smo = read_esipy(path)
   elif source == 'esipy':
      Smo = read_esipy(path)
   elif source == 'aimall':
      Smo = read_aimall(path)
   return Smo

def read_aimall(path='.'):
    Smo_alpha, Smo_beta = [], []
    Smo = []
    count = 0
    shape_Smo_alpha = 0
    indices = []
    start_string = 'The Atomic Overlap Matrix'

    if not os.path.exists(path):
        raise ValueError(f"The provided path '{path}' does not exist.")

    int_files = [intfile for intfile in os.listdir(path) if intfile.endswith('.int') and os.path.isfile(os.path.join(path, intfile))]
    files_ordered = sorted(int_files, key=lambda x: int(re.search(r'\d+', x).group()))

    for intfile in files_ordered:
        intfile_path = os.path.join(path, intfile)
        with open(intfile_path, 'r') as f:
            matrix_lines = []
            matrix_size = 1
            for num, line in enumerate(f, 1):

                # We first get the number of shape of the alpha-alpha matrix
                if 'First Beta MO' in line:
                    shape_Smo_alpha = int(line.split()[-1]) - 1

                if start_string in line:
                    next(f)
                    calcinfo = next(f)
                    next(f)
                    while True:
                        line = next(f).strip()
                        if not line:
                            break
                        matrix_lines.extend([float(num) for num in line.split()])
                        matrix_size += 1

                    matrix_size = int(np.sqrt(2 * len(matrix_lines) + 1/4) - 1/2)
                    lower_tri_matrix = np.zeros((matrix_size, matrix_size))
                    lower_tri_matrix[np.tril_indices(matrix_size)] = matrix_lines

                    matrix = lower_tri_matrix + lower_tri_matrix.T - np.diag(lower_tri_matrix.diagonal())

                    if 'Restricted' in calcinfo:
                       Smo.append(matrix)

                    if 'Unrestricted' in calcinfo and shape_Smo_alpha is not None:
                       SCR_alpha = matrix[:shape_Smo_alpha, :shape_Smo_alpha]
                       SCR_beta = matrix[shape_Smo_alpha:, shape_Smo_alpha:]
                       Smo_alpha.append(SCR_alpha)
                       Smo_beta.append(SCR_beta)


        numbers = [int(num) for num in re.findall(r'\d+', intfile)]
        indices.extend(numbers)

    if 'Restricted' in calcinfo:
       return Smo
    elif 'Unrestricted' in calcinfo:
       return [Smo_alpha, Smo_beta]

def read_esipy(path='.'):
    Smo = []
    Smo_alpha, Smo_beta = [], []
    count = 0
    indices = []
    start_string = 'The Atomic Overlap Matrix'

    if not os.path.exists(path):
        raise ValueError(f"The provided path '{path}' does not exist.")

    int_files = [intfile for intfile in os.listdir(path) if intfile.endswith('.int') and os.path.isfile(os.path.join(path, intfile))]
    files_ordered = sorted(int_files, key=lambda x: int(re.search(r'\d+', x).group()))

    for intfile in files_ordered:
        intfile_path = os.path.join(path, intfile)
        with open(intfile_path, 'r') as f:
            matrix_lines = []
            matrix_size = 1
            for line in f:
                if start_string in line:
                    next(f)
                    calcinfo = next(f)
                    next(f)
                    while True:
                        line = next(f).strip()
                        if not line:
                            break
                        matrix_lines.extend([float(num) for num in line.split()])

                    matrix_size = int(np.sqrt(2 * len(matrix_lines) + 1/4) - 1/2)
                    lower_tri_matrix = np.zeros((matrix_size, matrix_size))
                    lower_tri_matrix[np.tril_indices(matrix_size)] = matrix_lines

                    matrix = lower_tri_matrix + lower_tri_matrix.T - np.diag(lower_tri_matrix.diagonal())

                    if 'Restricted' in calcinfo:
                     Smo.append(matrix)

                    if 'Unrestricted' in calcinfo:

                       # We count how many zeros are there. Will be the shape of the alpha-alpha matrix
                       shape_Smo_alpha = 0
                       for num in matrix_lines:
                          if num == 0.0:
                             shape_Smo_alpha += 1
                          elif shape_Smo_alpha > 0:
                             break

                       SCR_alpha = matrix[:shape_Smo_alpha, :shape_Smo_alpha]
                       SCR_beta = matrix[shape_Smo_alpha:, shape_Smo_alpha:]
                       Smo_alpha.append(SCR_alpha)
                       Smo_beta.append(SCR_beta)


    numbers = [int(num) for num in re.findall(r'\d+', intfile)]
    indices.extend(numbers)

    if 'Restricted' in calcinfo:
       return Smo
    elif 'Unrestricted' in calcinfo:
       return [Smo_alpha, Smo_beta]

import numpy as np
import os
import re
from esi import rest_or_unrest

########### WRITING THE INPUT FOR THE ESI-3D CODE FROM THE AOMS ###########

def write_int(mol, mf, molname, Smo, ring=None, partition=None):
    """Writes the AOMs generated from the make_aoms() function as an input for the ESI-3D code.

    Arguments:
       mol: an instance of SCF class
          Mole class and helper functions to handle parameters and attributes for GTO integrals.

       mf: an instance of SCF class
          mf object holds all parameters to control SCF.

       molname: string
          A string containing the name of the input. Will be displayed in the output directories and files.

       Smo: list of matrices or string
          Atomic Overlap Matrices (AOMs) in the MO basis generated from the make_aoms() function.
          Can also be a string with the name of the file or the path where the AOMS have been saved.

       rings: list
          Contains a list of the indices of the atoms in the ring connectivity for the aromaticity calculations.

       partition: string
          Type of desired atom-in-molecule partition scheme. Options are 'mulliken', lowdin', 'meta_lowdin', 'nao' and 'iao'.

    Generates:
       A directory named 'molname'_'partition'.
       A file for each atom containing its AOM, readable for the ESI-3D code.
       A generalized input to the ESI-3D code, as 'molname'.bad. If no ring is specified, none will be displayed.
       A file 'molname'.titles containing the names of the generated .int files.
    """

    if isinstance(Smo, str):
        Smo = load_aoms(Smo)

    wf = rest_or_unrest(Smo)

    # Obtaining information for the files

    symbols = [mol.atom_symbol(i) for i in range(mol.natm)]
    atom_numbers = [i + 1 for i in range(mol.natm)]
    charge = mol.atom_charges()
    if wf == "unrest":
        nocc_alpha = mf.mo_occ[0].astype(int)
        nocc_beta = mf.mo_occ[1].astype(int)
        occ_coeff_beta = mf.mo_coeff[0][:, : nocc_beta.sum()]
        nalpha = [(charge + np.trace(aom_alpha)) / 2 for aom_alpha in Smo[0]]
        nbeta = [(charge + np.trace(aom_beta)) / 2 for aom_beta in Smo[1]]

        Smos = []
        fill = np.zeros((nocc_beta.sum(), nocc_alpha.sum()))
        for i in range(mol.natm):
            left = np.vstack((Smo[0][i], fill))
            right = np.vstack((fill.T, Smo[1][i]))
            matrix = np.hstack((left, right))
            Smos.append(matrix)

    else:
        nalpha = nbeta = [(charge + np.trace(aom)) / 2 for aom in Smo]

    # Creating a new directory for the calculation

    if partition == "mulliken":
        shortpart = "mul"
    elif partition == "lowdin":
        shortpart = "low"
    elif partition == "meta_lowdin":
        shortpart = "metalow"
    elif partition == "nao":
        shortpart = "nao"
    elif partition == "iao":
        shortpart = "iao"
    else:
        raise NameError("Hilbert-space scheme not available")

    new_dir_name = molname + "_" + shortpart
    titles = [symbols[i] + str(atom_numbers[i]) + shortpart for i in range(mol.natm) ]  # Setting the title of the files
    new_dir_path = os.path.join(os.getcwd(), new_dir_name)
    os.makedirs(new_dir_path, exist_ok=True)

    # Creating and writing the atomic .int files

    for i, item in enumerate(titles):
        with open(os.path.join(new_dir_path, item + ".int"), "w+") as f:
            f.write(" Created by ESIpy\n")
            if partition == "mulliken":
                f.write(" Using Mulliken atomic definition\n")
            elif partition == "lowdin":
                f.write(" Using Lowdin atomic definition\n")
            elif partition == "meta_lowdin":
                f.write(" Using Meta-Lowdin atomic definition\n")
            elif partition == "nao":
                f.write(" Using NAO atomic definition\n")
            elif partition == "iao":
                f.write(" Using IAO atomic definition\n")
            f.write(" Single-determinant wave function\n")
            if wf == "unrest" or wf == "rest":
                f.write(" Molecular SCF ENERGY (AU)  =       {:.11f}\n\n".format(mf.e_tot))
            else:
                f.write(" Molecular SCF ENERGY (AU)  =       \n\n")
            f.write(" INTEGRATION IS OVER ATOM  {}    {}\n".format(symbols[i], i + 1))
            f.write(" RESULTS OF THE INTEGRATION\n")
            if wf == "unrest":
                f.write("              N   {:.14E}    NET CHARGE {:.14E}\n".format(1, 1))
            else:
                f.write("              N   {:.14E}    NET CHARGE {:.14E}\n".format( 2 * np.trace(Smo[i]),
                        round(charge[i] - 2 * np.trace(Smo[i]), 14)))
            f.write("              G\n")
            f.write("              K   1.00000000000000E+01        E(ATOM)  1.00000000000000E+00\n")
            f.write("              L   1.00000000000000E+01\n\n")

            if wf == "unrest":
                f.write("\n The Atomic Overlap Matrix:\n\nUnrestricted\n\n")
                if partition == "mulliken":
                    f.write("  \n".join([ "  ".join(["{:.16E}".format(num, 16) for num in row])
                                for row in Smos[i] ]) + "\n")
                else:
                    f.write("\n".join(["  ".join([( "{:.16E}".format(Smos[i][j][k]) if j >= k else "")
                                        for k in range(len(Smos[i][j])) ]) for j in range(len(Smos[i])) ]) + "\n")
            else:
                f.write("\n          The Atomic Overlap Matrix\n\nRestricted Closed-Shell Wavefunction\n\n  ")
                if partition == "mulliken":
                    f.write("  \n".join(["  ".join(["{:.16E}".format(num, 16) for num in row]) for row in Smo[i] ]) + "\n")
                else:
                    f.write("\n".join(["  ".join([( "{:.16E}".format(Smo[i][j][k], 16) if j >= k else "")
                                        for k in range(len(Smo[i][j])) ]) for j in range(len(Smo[i])) ]) + "\n")
            f.write("\n                     ALPHA ELECTRONS (NA) {:E}\n".format(nalpha[i][0], 14))
            f.write("                      BETA ELECTRONS (NB) {:E}\n\n".format( nbeta[i][0], 14))
            f.write(" NORMAL TERMINATION OF PROAIMV")
            f.close()

    # Writing the file containing the title of the atomic .int files

    with open(os.path.join(new_dir_path, molname + shortpart + ".files"), "w") as f:
        for i in titles:
            f.write(i + ".int\n")
        f.close()

    # Creating the input for the ESI-3D code
    filename = os.path.join(new_dir_path, molname + ".bad")
    with open(filename, "w") as f:
        f.write("$TITLE\n")
        f.write(molname + "\n")
        f.write("$TYPE\n")
        if wf == "unrest":
            f.write("uhf\n{}\n".format(mol.nelec[0] + 1))
        else:
            f.write("hf\n")
        if len(ring) > 12:
            f.write("$NOMCI\n")
        f.write("$RING\n")
        if ring is not None:
            if isinstance(ring[0], int):  # If only one ring is specified
                f.write("1\n{}\n".format(len(ring)))
                f.write(" ".join(str(value) for value in ring))
                f.write("\n")
            else:
                f.write("{}\n".format(len(ring)))  # If two or more rings are specified as a list of lists
                for sublist in ring:
                    f.write(str(len(sublist)) + "\n")
                    f.write(" ".join(str(value) for value in sublist))
                    f.write("\n")
        else:
            f.write("\n")  # No ring specified, write it manually
        f.write("$ATOMS\n")
        f.write(str(mol.natm) + "\n")
        for title in titles:
            f.write(title + ".int\n")
        f.write("$BASIS\n")
        if wf == "unrest":
            f.write(str(int(np.shape(Smo[0])[1]) + int(np.shape(Smo[1])[1])) + "\n")
        elif wf == "unrest":
            f.write(str(int(np.shape(Smo)[1])) + "\n")
        else:
            f.write(str(np.shape(Smo)[1]) + "\n")
        f.write("$AV1245\n")
        f.write("$FULLOUT\n")
        if partition == "mulliken":
            f.write("$MULLIKEN\n")
