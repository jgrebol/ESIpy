import os
import re
import numpy as np
from esipy.tools import wf_type, format_short_partition, load_file


def read_aoms(path='.'):
    """
    Reads the AOMs from ESIpy's writeaoms() method or from an AIMAll calculation.

    :param path: Path of the directory of the files.
    :type path: str

    :returns: The AOMs in ESIpy format stored in 'ESI.aom'.
    :rtype: list
    """
    aom = []
    aom_alpha, aom_beta = [], []
    start_string = 'The Atomic Overlap Matrix'
    mul = False

    working_dir = os.getcwd()

    if path == '.':
        path = working_dir
    else:
        path = os.path.join(working_dir, path)

    ints = [intfile for intfile in os.listdir(path) if
            intfile.endswith('.int') and os.path.isfile(os.path.join(path, intfile))]
    ordered = sorted(ints, key=lambda x: int(re.search(r'\d+', x).group()))

    if ordered == []:
        raise ValueError(f"No *.int files found in the directory '{path}'.")

    first_file = os.path.join(path, ordered[0])
    with open(first_file, 'r') as f:
        for line in f:
            if "Restricted, closed" in line or "Restricted Closed" in line:
                wf = "rest"
                break
            elif "Unrestricted" in line:
                wf = "unrest"
                break
            elif "Restricted, natural" in line:
                wf = "no"
                break
        else:
            raise ValueError("Wavefunction type could not be determined.")

    if wf == "no":
        occs = read_occs(first_file)

    for intfile in ordered:
        intfile_path = os.path.join(path, intfile)
        with open(intfile_path, 'r') as f:
            mat_lines = []
            for line in f:
                if "Mulliken" in line:
                    mul = True
                if start_string in line:
                    for _ in range(3):
                        next(f)
                    while True:
                        line = next(f).strip()
                        if not line:
                            break
                        mat_lines.extend([float(num) for num in line.split()])
                        print(len(mat_lines))

                    # We first get the number of shape of the alpha-alpha matrix
                    print(mat_lines)
                    print(len(mat_lines))
                    na, nb = read_orbs(intfile_path)
                    nt = na + nb
                    print(na, nb, nt)

                    # Mulliken works on non-symmetric, square AOMs
                    if na != nb:
                        shape = nt
                    else:
                        shape = na

                    if mul:
                        matrix = np.array(mat_lines).reshape((shape, shape))
                    # Symmetric AOMs work on lower-triangular matrices
                    else:
                        low_matrix = np.zeros((shape, shape))
                        low_matrix[np.tril_indices(shape)] = mat_lines
                        matrix = low_matrix + low_matrix.T - np.diag(low_matrix.diagonal())

                    if wf == "rest" or wf == "no":
                        aom.append(matrix)

                    if wf == "unrest":
                        SCR_alpha = matrix[:na, :na]
                        SCR_beta = matrix[na:, na:]
                        aom_alpha.append(SCR_alpha)
                        aom_beta.append(SCR_beta)

    if wf == "rest":
        return aom
    elif wf == "unrest":
        return [aom_alpha, aom_beta]
    elif wf == "no":
        return [aom, occs]
    else:
        raise ValueError("Could not read the AOMs")


########### WRITING THE INPUT FOR THE ESI-3D CODE FROM THE AOMS ###########

def write_aoms(mol, mf, name, aom, ring=[], partition=None):
    """
        Writes the input for the ESI-3D code from the AOMs.

    :param mol: Molecule object from PySCF.
    :type mol: PySCF instance
    :param mf: Calculation object from PySCF.
    :type mf: PySCF instance
    :param name: Name of the calculation.
    :type name: str
    :param aom: Concatenated list of Atomic Overlap Matrices (AOMs) in the MO basis.
    :type aom: list
    :param ring: Connectivity of the atoms in the ring. Can be more than one ring as a list of lists.
    :type ring: list of int, optional
    :param partition: Partition scheme for the AOMs. Options are "mulliken", "lowdin", "meta_lowdin", "nao", "iao".
    :type partition: str, optional

    :returns: None

    :generates:
        - A '_atomicfiles/' directory with all the files created.
        - A '.int' file for each atom with its corresponding AOM.
        - A 'name.files' file with a list of the names of the '.int' files.
        - A 'name.bad' file with a standard input for the ESI-3D code.
        - For Natural Orbitals, a 'name.wfx' file with the occupancies for the ESI-3D code.
    """
    from copy import deepcopy

    if isinstance(aom, str):
        aom = load_file(aom)
    if ring is None:
        pass
    elif isinstance(ring[0], int):
        ring = [ring]

    wf = wf_type(aom)
    if wf == "no":
        aom, occ = aom  # Separating AOMs and orbital occupations

    # Obtaining information for the files

    if wf == "unrest":
        nalpha = [np.trace(aom_alpha) for aom_alpha in aom[0]]
        nbeta = [np.trace(aom_beta) for aom_beta in aom[1]]

    elif wf == "rest":
        nalpha = nbeta = [np.trace(aom) for aom in aom]

    elif wf == "no":
        nalpha = nbeta = [float(np.trace(np.dot(occ, aom))) for aom in aom]

    # Creating a new directory for the calculation

    atom_numbers = [i + 1 for i in range(mol.natm)]
    symbols = [mol.atom_symbol(i) for i in range(mol.natm)]
    fragidx = mol.natm + 1
    fragmap = {}
    dofrag = False

    ringcopy = deepcopy(ring)

    # Mark fragments in the order they are defined
    if ring:
        for i, sublist in enumerate(ringcopy):
            for j, element in enumerate(sublist):
                if isinstance(element, set):
                    dofrag = True
                    fragmap[tuple(element)] = fragidx
                    sublist[j] = fragidx
                    fragidx += 1


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
                occ = np.diag([1.0] * (len(aom[0][i]) + len(aom[1][i])))
                f.write("\n The Atomic Overlap Matrix:\n\n Unrestricted\n\n")
                alpha_size = len(aom[0][i])
                beta_size = len(aom[1][i])
                zeros = np.zeros((beta_size, alpha_size))
                fill = np.vstack((aom[0][i], zeros))
                fill = np.hstack((fill, np.vstack((zeros.T, aom[1][i]))))
                if partition == "mulliken":
                    for j in range(alpha_size + beta_size):
                        for k in range(alpha_size + beta_size):
                            f.write("{:.16E}  ".format(fill[j][k]))
                        f.write("\n")
                else:
                    for j in range(alpha_size + beta_size):
                        for k in range(alpha_size + beta_size):
                            if k <= j:
                                f.write("{:.16E}  ".format(fill[j][k]))
                        f.write("\n")
            elif wf == "rest":
                occ = np.diag([2.0] * len(aom[i]))
                f.write("\n The Atomic Overlap Matrix:\n\n Restricted, closed-shell\n\n  ")
                if partition == "mulliken":
                    for j in range(len(aom[i])):
                        for k in range(len(aom[i])):
                            f.write("{:.16E}  ".format(aom[i][j][k]))
                        f.write("\n")
                else:
                    for j in range(len(aom[i])):
                        for k in range(len(aom[i])):
                            if k <= j:
                                f.write("{:.16E}  ".format(aom[i][j][k]))
                        f.write("\n")
            elif wf == "no":
                f.write("\n The Atomic Overlap Matrix:\n\n Restricted, natural orbital wavefunction\n\n  ")
                if partition == "mulliken":
                    for j in range(len(aom[i])):
                        for k in range(len(aom[i])):
                            f.write("{:.16E}  ".format(aom[i][j][k]))
                        f.write("\n")
                else:
                    for j in range(len(aom[i])):
                        for k in range(len(aom[i])):
                            if k <= j:
                                f.write("{:.16E}  ".format(aom[i][j][k]))
                        f.write("\n")

            f.write("\n\n")
            f.write("Molecular Orbital (MO) Data:\n")
            f.write("---------------------------\n")
            f.write("  MO# i = ith MO in AIMAll (internal and output) order\n")
            f.write("  WMO#(i) = MO# i in wavefunction file order\n")
            f.write("  Occ_MO(i) = Occupancy of ith MO for Molecule\n")
            f.write("  Spin_MO(i) = Spin Type of ith MO\n")
            f.write("  Occ_MO(A,i) = Contribution of Atom A to Occ_MO(i)\n")
            f.write("  %Occ_MO(A,i) = 100 * Occ_MO(A,i) / Occ_MO(i)\n")
            f.write("  %N_MO(A,i) = 100 * Occ_MO(A,i) / N(A)\n")
            f.write(
                "---------------------------------------------------------------------------------------------------\n")
            f.write(
                "    MO# i    WMO#(i)     Occ_MO(i)     Spin_MO(i)     Occ_MO(A,i)     %Occ_MO(A,i)     %N_MO(A,i)\n")
            f.write(
                "---------------------------------------------------------------------------------------------------\n")

            if wf == "unrest":
                for j, occup in enumerate(np.diag(occ[beta_size:])):
                    f.write(
                        f"       {j + 1:<8}{j + 1:<12}{float(1.):<15.10f}{'Alpha':<15}{0.0:<15.10f}{0.0:<15.10f}{0.0:<15.10f}\n"
                    )
                for j, occup in enumerate(np.diag(occ[:beta_size])):
                    f.write(
                        f"       {j + 1 + alpha_size:<8}{j + 1 + alpha_size:<12}{float(occup):<15.10f}{'Beta':<15}{0.0:<15.10f}{0.0:<15.10f}{0.0:<15.10f}\n"
                    )
            else:
                for j, occup in enumerate(np.diag(occ)):
                    f.write(
                        f"       {j + 1:<8}{j + 1:<12}{float(occup):<15.10f}{'Alpha,Beta':<15}{0.0:<15.10f}{0.0:<15.10f}{0.0:<15.10f}\n"
                    )
            f.write("\n Alpha electrons (NAlpha)                        {:.10E}".format(nalpha[i]))
            f.write("\n Beta electrons (NBeta)                          {:.10E}\n".format(nbeta[i]))
            f.write(" NORMAL TERMINATION OF PROAIMV")
            f.close()

    # Writing the file containing the title of the atomic .int files
    with open(os.path.join(new_dir_path, name + shortpart + ".files"), "w") as f:
        for i in titles:
            f.write(i + ".int\n")
        f.close()

    # Writing the file containing the title of the atomic .int files
    domci = False
    if ring:
        if isinstance(ring[0], int):
            ring = [ring]
        for r in ring:
            if len(r) < 10:
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
            if dofrag:
                f.write("$FRAGMENTS\n")
                f.write(f"{len(fragmap)}\n")
                for fragatm in fragmap.keys():
                    f.write(f"{len(fragatm)}\n")
                    f.write(" ".join(str(value) for value in fragatm))
                    f.write("\n")
            if ring:
                f.write("$RING\n")
                f.write("{}\n".format(len(ring)))  # If two or more rings are specified as a list of lists
                for sublist in ring:
                    f.write(str(len(sublist)) + "\n")
                    mapped = [fragmap.get(tuple(x), x) if isinstance(x, set) else x for x in sublist]
                    f.write(" ".join(str(value) for value in mapped))
                    f.write("\n")
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
            if dofrag:
                f.write("$FRAGMENTS\n")
                f.write(f"{len(fragmap)}\n")
                for fragatm in fragmap.keys():
                    f.write(f"{len(fragatm)}\n")
                    f.write(" ".join(str(value) for value in fragatm))
                    f.write("\n")
            if ring:
                f.write("$RING\n")
                f.write("{}\n".format(len(ring)))  # If two or more rings are specified as a list of lists
                for sublist in ring:
                    f.write(str(len(sublist)) + "\n")
                    mapped = [fragmap.get(tuple(x), x) if isinstance(x, set) else x for x in sublist]
                    f.write(" ".join(str(value) for value in mapped))
                    f.write("\n")
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

def read_molinfo(path):
    """
    Reads all *.int files from a directory and builds the molinfo dictionary.

    :param path: Path to the directory containing *.int files.
    :type path: str

    :returns: A dictionary containing molecular information.
    :rtype: dict
    """
    molinfo = {
        "method": "Not specified",
        "basisset": "Not specified",
        "geom": None,
        "partition": "qtaim",
    }

    symbs, atm_nums = [], []
    found_energy = False
    ints = [intfile for intfile in os.listdir(path) if
            intfile.endswith('.int') and os.path.isfile(os.path.join(path, intfile))]
    int_files = sorted(ints, key=lambda x: int(re.search(r'\d+', x).group()))

    if not int_files:
        raise FileNotFoundError(f"No *.int files found in the directory '{path}'.")

    molinfo["partition"] = get_partition(path, int_files[0])
    for int_file in int_files:
        with open(os.path.join(path, int_file), "r") as file:
            lines = file.readlines()

        for line in lines:
            if "Integration is" in line or "INTEGRATION IS" in line:
                parts = re.findall(r'[A-Za-z]+|\d+', line.split()[-1].strip())
                symbs.append(parts[-2])
                atm_nums.append(parts[-1])
            if "Restricted, closed-shell" in line or "Restricted Closed-Shell":
                molinfo["calctype"] = "RHF"
            if "Unrestricted" in line:
                molinfo["calctype"] = "UHF"
            elif "Restricted, natural" in line:
                molinfo["calctype"] = "NO"

            if "The molecular energy from the wf" in line or "ENERGY" in line and not found_energy:
                molinfo["energy"] = float(line.split()[-1])
                found_energy = True
    molinfo["symbols"] = symbs
    molinfo["atom_numbers"] = atm_nums
    molinfo["geom"] = read_wfx_info(path) if any(f.endswith('.wfx') for f in os.listdir(path)) else None

    return molinfo

def read_wfx_info(path):
    """
    Searches for a .wfx file in the given path or the previous path, reads the coordinates and charges,
    and stores them in molinfo["geom"].

    :param path: Path to search for the .wfx file.
    :type path: str
    :param molinfo: Dictionary to store molecular information.
    :type molinfo: dict
    :returns: NumPy array with <symbol> <x> <y> <z>.
    :rtype: numpy.ndarray
    """
    # Look for a .wfx file in the current path
    wfx_files = [f for f in os.listdir(path) if f.endswith('.wfx')]
    if not wfx_files or len(wfx_files) > 1:
        # Try the previous path if no .wfx file is found
        previous_path = os.path.dirname(path)
        wfx_files = [f for f in os.listdir(previous_path) if f.endswith('.wfx')]
        if not wfx_files or len(wfx_files) > 1:
            print(" | Could not find .wfx file in", path, "\n | or", previous_path)
            return None

        path = previous_path

    wfx_file = os.path.join(path, wfx_files[0])
    with open(wfx_file, 'r') as file:
        lines = file.readlines()

    start_coords = False
    coordinates = []
    for line in lines:
        if "<Nuclear Cartesian Coordinates>" in line:
            start_coords = True
            continue
        if "</Nuclear Cartesian Coordinates>" in line:
            break
        if start_coords:
            parts = line.split()
            if len(parts) == 3:
                coordinates.append([float(x) for x in parts])

    if not coordinates:
        raise ValueError("No coordinates found in the .wfx file.")

    # Combine symbols and coordinates into a NumPy array
    geom = np.array([coordinates[i] for i in range(len(coordinates))], dtype=object)

    # Store the coordinates in molinfo["geom"]
    return geom


def get_partition(path, int_file):
    """
    Extracts the partition type from the given file.

    :param path: Directory path containing the file.
    :param int_file: File name to read.
    :return: Partition type as a string.
    """
    with open(os.path.join(path, int_file), "r") as file:
        lines = file.readlines()
    if "AIMInt" in lines[0]:
        return "qtaim"
    elif "ESIpy" in lines[0]:
        return lines[1].split()[1].strip()
    return None

def read_occs(file_path):
    """
    Extracts the NO coefficients from a Restricted, natural orbitals AIMAll calculation.

    :param file_path: Path to the file to process.
    :type file_path: str
    :return: List of extracted Occ_MO(i) values.
    :rtype: list of float
    """
    occ_mo_values = []
    start_processing = False

    with open(file_path, 'r') as file:
        for line in file:
            # Start processing after the header line
            if "Molecular Orbital (MO) Data" in line:
                start_processing = True
                for _ in range(10):
                    next(file)
                continue

            # Stop processing at a blank line
            if start_processing and not line.strip():
                break

            # Extract the third column if processing has started
            if start_processing:
                columns = line.split()
                if len(columns) >= 3:
                    occ_mo_values.append(float(columns[2]))

    return np.diag(occ_mo_values)

def read_orbs(file_path):
    """
    Extracts the NO coefficients from a Restricted, natural orbitals AIMAll calculation.

    :param file_path: Path to the file to process.
    :type file_path: str
    :return: List of extracted Occ_MO(i) values.
    :rtype: list of float
    """
    nalpha, nbeta = 0, 0
    start_processing = False

    with open(file_path, 'r') as file:
        for line in file:
            # Start processing after the header line
            if "Molecular Orbital (MO) Data" in line:
                start_processing = True
                for _ in range(10):
                    next(file)
                continue

            # Stop processing at a blank line
            if start_processing and not line.strip():
                break

            if start_processing:
                columns = line.split()
                if len(columns) >= 3:
                    if columns[3] == "Alpha":
                        nalpha += 1
                    elif columns[3] == "Beta":
                        nbeta += 1
                    elif columns[3] == "Alpha,Beta":
                        nalpha += 1
                        nbeta += 1
                    else:
                       raise ValueError("Invalid spin type in the input file.")
    return nalpha, nbeta