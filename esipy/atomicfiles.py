import os
import re
import numpy as np
from esipy.tools import wf_type

def read_aimall(path='.'):
    """
    Reads the AOMs from AIMAll.
    Args:
        path: Path of the directory of the files.

    Returns:
        The AOMs in ESIpy format stored in 'ESI.Smo'.
    """
    Smo_alpha, Smo_beta = [], []
    Smo = []
    shape_Smo_alpha = 0
    start_string = 'The Atomic Overlap Matrix'
    mul = False

    if not os.path.exists(path):
        raise ValueError(f"The provided path '{path}' does not exist.")

    ints = [intfile for intfile in os.listdir(path) if
                 intfile.endswith('.int') and os.path.isfile(os.path.join(path, intfile))]
    ordered = sorted(ints, key=lambda x: int(re.search(r'\d+', x).group()))

    for intfile in ordered:
        intfile_path = os.path.join(path, intfile)
        with open(intfile_path, 'r') as f:
            print("Doing for file", intfile)
            mat_lines = []
            mat_size = 1
            for num, line in enumerate(f, 1):

                if "Mulliken" in line:
                    mul = True
                # We first get the number of shape of the alpha-alpha matrix
                if 'First Beta MO' in line:
                    shape_Smo_alpha = int(line.split()[-1]) - 1

                if start_string in line:
                    next(f)
                    calcinfo = next(f)
                    print(calcinfo)
                    next(f)
                    while True:
                        line = next(f).strip()
                        if not line:
                            break
                        mat_lines.extend([float(num) for num in line.split()])
                        mat_size += 1


                    if 'Restricted' in calcinfo:
                        Smo.append(matrix)

                    if 'Unrestricted' in calcinfo and shape_Smo_alpha is not None:
                        SCR_alpha = matrix[:shape_Smo_alpha, :shape_Smo_alpha]
                        SCR_beta = matrix[shape_Smo_alpha:, shape_Smo_alpha:]
                        Smo_alpha.append(SCR_alpha)
                        Smo_beta.append(SCR_beta)

    if 'Restricted' in calcinfo:
        return Smo
    elif 'Unrestricted' in calcinfo:
        return [Smo_alpha, Smo_beta]


def read_aoms(path='.'):
    """
    Reads the AOMs from ESIpy's writeaoms() method.
    Args:
        path: Path of the directory of the files.

    Returns:
        The AOMs in ESIpy format stored in 'ESI.Smo'.
    """
    Smo = []
    Smo_alpha, Smo_beta = [], []
    start_string = 'The Atomic Overlap Matrix'
    mul = False

    path = os.path.join(os.getcwd(), path)
    if not os.path.exists(path):
        raise ValueError(f"The provided path '{path}' does not exist.")

    ints = [intfile for intfile in os.listdir(path) if
                 intfile.endswith('.int') and os.path.isfile(os.path.join(path, intfile))]
    ordered = sorted(ints, key=lambda x: int(re.search(r'\d+', x).group()))

    for intfile in ordered:
        intfile_path = os.path.join(path, intfile)
        with open(intfile_path, 'r') as f:
            mat_lines = []
            for line in f:
                if "Mulliken" in line:
                    mul = True
                if start_string in line:
                    next(f)
                    calcinfo = next(f)
                    next(f)
                    while True:
                        line = next(f).strip()
                        if not line:
                            break
                        mat_lines.extend([float(num) for num in line.split()])

                    # Mulliken works on non-symmetric AOMs
                    if mul:
                        mat_size = int(np.sqrt(len(mat_lines)))
                        matrix = np.array(mat_lines).reshape((mat_size, mat_size))
                    # Symmetric AOMs work on lower-triangular matrices
                    else:
                        mat_size = int(np.sqrt(2 * len(mat_lines) + 1 / 4) - 1 / 2)
                        low_matrix = np.zeros((mat_size, mat_size))
                        low_matrix[np.tril_indices(mat_size)] = mat_lines
                        matrix = low_matrix + low_matrix.T - np.diag(low_matrix.diagonal())


                    # We first get the number of shape of the alpha-alpha matrix
                    if 'First Beta MO' in line:
                        shape_Smo_alpha = int(line.split()[-1]) - 1
                    else:
                        shape_Smo_alpha = 0
                        for num in mat_lines:
                            if num == 0.0:
                                shape_Smo_alpha += 1
                            elif shape_Smo_alpha > 0:
                                break

                    if 'Restricted' in calcinfo:
                        Smo.append(matrix)

                    if 'Unrestricted' in calcinfo:

                        SCR_alpha = matrix[:shape_Smo_alpha, :shape_Smo_alpha]
                        SCR_beta = matrix[shape_Smo_alpha:, shape_Smo_alpha:]
                        Smo_alpha.append(SCR_alpha)
                        Smo_beta.append(SCR_beta)

    if 'Restricted' in calcinfo:
        return Smo
    elif 'Unrestricted' in calcinfo:
        return [Smo_alpha, Smo_beta]


########### WRITING THE INPUT FOR THE ESI-3D CODE FROM THE AOMS ###########

def write_aoms(mol, mf, name, Smo, ring=None, partition=None):
    """Writes the AOMs as an input for the ESI-3D code.

    Arguments:
       mol (SCF instance, optional, default: None):
           PySCF's Mole class and helper functions to handle parameters and attributes for GTO integrals.

       mf (SCF instance, optional, default: None):
           PySCF's object holds all parameters to control SCF.

       name (string):
           A string containing the name of the calculation.

       Smo (list of matrices or str):
            Specifies the Atomic Overlap Matrices (AOMs) in the MO basis. This can either be a list of matrices generated from the `make_aoms()` function or a string with the filename/path where the AOMs are saved.

       rings (list or list of lists of int):
            Contains the indices defining the ring connectivity of a system. Can contain several rings as a list of lists.

       partition (str, optional, default: None):
           Specifies the atom-in-molecule partition scheme. Options include 'mulliken', 'lowdin', 'meta_lowdin', 'nao', and 'iao'.

    Generates:
       A directory named 'name'_'partition'.
       A file for each atom, '.int', containing its AOM, readable for the ESI-3D code.
       A generalized input to the ESI-3D code, as 'name'.bad. If no ring is specified, none will be displayed.
       A file 'name'.titles containing the names of the generated .int files.
    """

    if isinstance(Smo, str):
        Smo = load_aoms(Smo)

    wf = wf_type(Smo)
    if wf == "no":
        Smo, occ = Smo # Separating AOMs and orbital occupations

    # Obtaining information for the files

    symbols = [mol.atom_symbol(i) for i in range(mol.natm)]
    atom_numbers = [i + 1 for i in range(mol.natm)]
    if wf == "unrest":
        nocc_alpha = mf.mo_occ[0].astype(int)
        nocc_beta = mf.mo_occ[1].astype(int)
        nalpha = [np.trace(aom_alpha) for aom_alpha in Smo[0]]
        nbeta = [np.trace(aom_beta) for aom_beta in Smo[1]]

        Smos = []
        fill = np.zeros((nocc_beta.sum(), nocc_alpha.sum()))
        for i in range(mol.natm):
            left = np.vstack((Smo[0][i], fill))
            right = np.vstack((fill.T, Smo[1][i]))
            matrix = np.hstack((left, right))
            Smos.append(matrix)

    else:
        nalpha = nbeta = [np.trace(aom) for aom in Smo]

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

    new_dir_name = name + "_" + shortpart
    symbols = [s.lower() for s in symbols]
    titles = [symbols[i] + str(atom_numbers[i]) for i in range(mol.natm)]  # Setting the title of the files
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
            f.write(" Molecular SCF ENERGY (AU)  =       {:.11f}\n\n".format(mf.e_tot))
            f.write(" INTEGRATION IS OVER ATOM  {}\n".format(titles[i]))
            f.write(" RESULTS OF THE INTEGRATION\n")
            if wf == "unrest":
                f.write("          N   {:.10E}    NET CHARGE 0.0000000000E+00\n".format(np.trace(Smo[0][i]) + np.trace(Smo[1][i])))
            elif wf == "rest":
                f.write("          N   {:.10E}    NET CHARGE 0.0000000000E+00\n".format(2 * np.trace(Smo[i])))
            else:
                f.write("          N   {:.10E}    NET CHARGE 0.0000000000E+00\n".format(np.trace(np.dot(occ, Smo[i]))))
            f.write("              G\n")
            f.write("              K   1.00000000000000E+01        E(ATOM)  1.00000000000000E+00\n")
            f.write("              L   0.00000000000000E+01\n\n")

            if wf == "unrest":
                f.write("\n The Atomic Overlap Matrix:\n\n Unrestricted\n\n")
                if partition == "mulliken":
                    f.write("  \n".join(["  ".join(["{:.16E}".format(num, 16) for num in row])
                                         for row in Smos[i]]) + "\n")
                else:
                    f.write("\n".join(["  ".join([("{:.16E}".format(Smos[i][j][k]) if j >= k else "")
                                                  for k in range(len(Smos[i][j]))]) for j in
                                       range(len(Smos[i]))]) + "\n")
            else:
                f.write("\n The Atomic Overlap Matrix:\n\n Restricted Closed-Shell Wavefunction\n\n  ")
                if partition == "mulliken":
                    f.write(
                        "  \n".join(["  ".join(["{:.16E}".format(num, 16) for num in row]) for row in Smo[i]]) + "\n")
                else:
                    f.write("\n".join(["  ".join([("{:.16E}".format(Smo[i][j][k], 16) if j >= k else "")
                                                  for k in range(len(Smo[i][j]))]) for j in range(len(Smo[i]))]) + "\n")
            f.write("\n Alpha electrons (NAlpha)                        {:.10E}".format(nalpha[i]))
            f.write("\n Beta electrons (NBeta)                          {:.10E}\n".format(nbeta[i]))
            f.write(" NORMAL TERMINATION OF PROAIMV")
            f.close()

    # Writing the file containing the title of the atomic .int files
    with open(os.path.join(new_dir_path, name + shortpart + ".files"), "w") as f:
        for i in titles:
            f.write(i + ".int\n")
        f.close()

    if wf == "no": # Will get information from ESIpy's wfx
        filename = os.path.join(new_dir_path, name + ".bad")
        with open(filename, "w") as f:
            f.write("$READWFN\n")
            f.write(name + ".wfx\n")
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
            f.write("$AV1245\n")
            f.write("$FULLOUT\n")
            if partition == "mulliken":
                f.write("$MULLIKEN\n")
            f.close()

    # Single-determinant input file
    else:
        # Creating the input for the ESI-3D code
        filename = os.path.join(new_dir_path, name + ".bad")
        with open(filename, "w") as f:
            f.write("$TITLE\n")
            f.write(name + "\n")
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
            else:
                f.write(str(np.shape(Smo)[1]) + "\n")
            f.write("$AV1245\n")
            f.write("$FULLOUT\n")
            f.write("$DEBUG\n")
            if partition == "mulliken":
                f.write("$MULLIKEN\n")
            f.close()

        # Creating custom .wfx with occupancies for the ESI-3D code
    if wf == "no":
        filename = os.path.join(new_dir_path, name + ".wfx")
        with open(filename, "w") as f:
            f.write('<Number of Nuclei>\n')
            f.write(f" {mol.natm}\n")
            f.write('</Number of Nuclei>\n')

            f.write('<Number of Occupied Molecular Orbitals>\n')
            f.write(f" {str(len(mf.mo_coeff))}\n")
            f.write('</Number of Occupied Molecular Orbitals>\n')

            f.write('<Number of Electrons>\n')
            f.write(f" {sum(mol.nelec)}\n")
            f.write('</Number of Electrons>\n')

            f.write('<Number of Core Electrons>\n')
            f.write(f" {0}\n")
            f.write('</Number of Core Electrons>\n')

            nalpha, nbeta = mol.nelec
            f.write('<Number of Alpha Electrons>\n')
            f.write(f" {nalpha}\n")
            f.write('</Number of Alpha Electrons>\n')
            f.write('<Number of Beta Electrons>\n')
            f.write(f" {nbeta}\n")
            f.write('</Number of Beta Electrons>\n')

            f.write("<Electronic Spin Multiplicity>\n")
            mult = 2 * mol.spin + 1
            f.write(f" {str(mult)}\n")
            f.write("</Electronic Spin Multiplicity>\n")

            f.write("<Nuclear Names>\n")
            for i in titles:
                f.write(f" {i.upper()}\n")
            f.write("</Nuclear Names>\n")

            f.write("<Nuclear Cartesian Coordinates>\n")
            for coord in mol.atom_coords():
                f.write(" {: .12e} {: .12e} {: .12e}\n".format(coord[0], coord[1], coord[2]))
            f.write("</Nuclear Cartesian Coordinates>\n")

            f.write("<Molecular Orbital Occupation Numbers>\n")
            for occupation in np.diag(occ):
                f.write(f"  {occupation:.12e}\n")
            f.write("</Molecular Orbital Occupation Numbers>\n")

            f.write("<Molecular Orbital Spin Types>\n")
            for i in range(0, len(Smo[0])):
                f.write(" Alpha and Beta\n")
            f.write("</Molecular Orbital Spin Types>\n")

            f.write("<Molecular Orbital Energies>\n")
            for i in range(0, len(Smo[0])):
                f.write("  0.000000000000e+00\n")
            f.write("</Molecular Orbital Energies>\n")

            f.close()
