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

