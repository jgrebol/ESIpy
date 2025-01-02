import os
import re

import numpy as np

from esipy.tools import wf_type, format_short_partition


def read_aoms(path='.'):
    """
    Reads the AOMs from ESIpy's writeaoms() method or from an AIMAll calculation.
    Args:
        path: Path of the directory of the files.

    Returns:
        The AOMs in ESIpy format stored in 'ESI.aom'.
    """
    aom = []
    aom_alpha, aom_beta = [], []
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

                    # Mulliken works on non-symmetric, square AOMs
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
                        shape_aom_alpha = int(line.split()[-1]) - 1
                    else:
                        shape_aom_alpha = 0
                        for num in mat_lines:
                            if num == 0.0:
                                shape_aom_alpha += 1
                            elif shape_aom_alpha > 0:
                                break

                    if 'Restricted' in calcinfo:
                        aom.append(matrix)

                    if 'Unrestricted' in calcinfo:
                        SCR_alpha = matrix[:shape_aom_alpha, :shape_aom_alpha]
                        SCR_beta = matrix[shape_aom_alpha:, shape_aom_alpha:]
                        aom_alpha.append(SCR_alpha)
                        aom_beta.append(SCR_beta)

    if 'Restricted' in calcinfo:
        return aom
    elif 'Unrestricted' in calcinfo:
        return [aom_alpha, aom_beta]


########### WRITING THE INPUT FOR THE ESI-3D CODE FROM THE AOMS ###########

def write_aoms(mol, mf, name, aom, ring=None, partition=None):
    """
    Writes the input for the ESI-3D code from the AOMs.
    Parameters:
        mol: PySCF instance
            Molecule object from PySCF.
        mf: PySCF instance
            Calculation object from PySCF.
        name: str
            Name of the calculation.
        aom: concatenated list
            Atomic Overlap Matrices (AOMs) in the MO basis.
        ring: list of int
            Connectivity of the atoms in the ring. Can be more than one ring as a list of lists.
        partition: str
            Partition scheme for the AOMs. Options are "mulliken", "lowdin", "meta_lowdin", "nao", "iao".

    Generates:
        - A '_atomicfiles/' directory with all the files created.
        - A '.int' file for each atom with its corresponding AOM.
        - A 'name.files' with a list of the names of the '.int' files.
        - A 'name.bad' with a standard input for the ESI-3D code.
        - For Natural Orbitals, a 'name.wfx' with the occupancies for the ESI-3D code.
    """

    if isinstance(aom, str):
        aom = load_aoms(aom)

    wf = wf_type(aom)
    if wf == "no":
        aom, occ = aom  # Separating AOMs and orbital occupations

    # Obtaining information for the files

    symbols = [mol.atom_symbol(i) for i in range(mol.natm)]
    atom_numbers = [i + 1 for i in range(mol.natm)]
    if wf == "unrest":
        nocc_alpha = mf.mo_occ[0].astype(int)
        nocc_beta = mf.mo_occ[1].astype(int)
        nalpha = [np.trace(aom_alpha) for aom_alpha in aom[0]]
        nbeta = [np.trace(aom_beta) for aom_beta in aom[1]]

        aoms = []
        fill = np.zeros((nocc_beta.sum(), nocc_alpha.sum()))
        for i in range(mol.natm):
            left = np.vstack((aom[0][i], fill))
            right = np.vstack((fill.T, aom[1][i]))
            matrix = np.hstack((left, right))
            aoms.append(matrix)

    else:
        nalpha = nbeta = [np.trace(aom) for aom in aom]

    # Creating a new directory for the calculation

    shortpart = format_short_partition(partition)

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
                f.write("          N   {:.10E}    NET CHARGE 0.0000000000E+00\n".format(
                    np.trace(aom[0][i]) + np.trace(aom[1][i])))
            elif wf == "rest":
                f.write("          N   {:.10E}    NET CHARGE 0.0000000000E+00\n".format(2 * np.trace(aom[i])))
            else:
                f.write("          N   {:.10E}    NET CHARGE 0.0000000000E+00\n".format(np.trace(np.dot(occ, aom[i]))))
            f.write("              G\n")
            f.write("              K   1.00000000000000E+01        E(ATOM)  1.00000000000000E+00\n")
            f.write("              L   0.00000000000000E+01\n\n")

            if wf == "unrest":
                f.write("\n The Atomic Overlap Matrix:\n\n Unrestricted\n\n")
                if partition == "mulliken":
                    f.write("  \n".join(["  ".join(["{:.16E}".format(num, 16) for num in row])
                                         for row in aoms[i]]) + "\n")
                else:
                    f.write("\n".join(["  ".join([("{:.16E}".format(aoms[i][j][k]) if j >= k else "")
                                                  for k in range(len(aoms[i][j]))]) for j in
                                       range(len(aoms[i]))]) + "\n")
            else:
                f.write("\n The Atomic Overlap Matrix:\n\n Restricted Closed-Shell Wavefunction\n\n  ")
                if partition == "mulliken":
                    f.write(
                        "  \n".join(["  ".join(["{:.16E}".format(num, 16) for num in row]) for row in aom[i]]) + "\n")
                else:
                    f.write("\n".join(["  ".join([("{:.16E}".format(aom[i][j][k], 16) if j >= k else "")
                                                  for k in range(len(aom[i][j]))]) for j in range(len(aom[i]))]) + "\n")
            f.write("\n Alpha electrons (NAlpha)                        {:.10E}".format(nalpha[i]))
            f.write("\n Beta electrons (NBeta)                          {:.10E}\n".format(nbeta[i]))
            f.write(" NORMAL TERMINATION OF PROAIMV")
            f.close()

    # Writing the file containing the title of the atomic .int files
    with open(os.path.join(new_dir_path, name + shortpart + ".files"), "w") as f:
        for i in titles:
            f.write(i + ".int\n")
        f.close()

    domci = False
    if isinstance(ring[0], int):
        ring = [ring]
    for r in ring:
        if len(r) < 12:
            domci = True

    # Single-determinant input file
    if wf == "rest" or wf == "unrest":
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
            if not domci:
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
                f.write(str(int(np.shape(aom[0])[1]) + int(np.shape(aom[1])[1])) + "\n")
            else:
                f.write(str(np.shape(aom)[1]) + "\n")
            f.write("$AV1245\n")
            f.write("$FULLOUT\n")
            if partition == "mulliken":
                f.write("$MULLIKEN\n")
            f.close()

    # Natural orbitals input file
    elif wf == "no":
        # Creating the input for the ESI-3D code
        filename = os.path.join(new_dir_path, name + ".bad")
        with open(filename, "w") as f:
            f.write("$READWFN\n")
            f.write(name + ".wfx\n")
            if not domci:
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

        # Creating a custom .wfx file for the ESI-3D code with the occupation numbers
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
            for i in range(0, len(aom[0])):
                f.write(" Alpha and Beta\n")
            f.write("</Molecular Orbital Spin Types>\n")

            f.write("<Molecular Orbital Energies>\n")
            for i in range(0, len(aom[0])):
                f.write("  0.000000000000e+00\n")
            f.write("</Molecular Orbital Energies>\n")

            f.close()
