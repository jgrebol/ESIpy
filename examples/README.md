# Examples

In this section we will go through the process of performing all the available calculations using ESIpy's functions based on the provided examples. The program is intended to be installed locally in computer clusters. To run the code from the terminal, generate the Python script or adapt those of the ```examples``` repository and run it as ```python code.py``` or ```python3 code.py```. To save the output as a file, use ```python code.py > code.esi```. Even though the ```.esi``` extension is not mandatory, we recommend using it. The test files have been generated from benzene and naphthalene at singlet (and triplet) level of theory, and are just set to showcase the capabilities of the code. We will go through the list of examples explaining and highlighting some key notes:

- example01: This should be the first calculation performed from ESIpy, in which one generates the Atomic Overlap Matrices in the MO basis ($\boldsymbol{S}^{\text{A}}$) from the ```make_aoms()``` function, and the molecular information from the ```mol_info()``` function. This ensures that the single-point calculation is not required anymore every time one wants to perform the aromaticity calculation, and allows for more flexibility with the code. We strongly suggest using the ```.aoms``` and ```.molinfo``` extensions for these files. Even though the main ```aromaticity()``` function only requires the AOMs and the ring indices for a minimal output, we suggest using the ```.molinfo``` for completion. 

> We strongly recommend using ```meta_lowdin```, ```nao``` and ```iao``` as the atomic partitions as they have shown to be highly basis-set independent and reliable. We introduce the five atomic partitions available at ESIpy in a for-loop scheme, although one partition can be introduced for each calculation. The computation time is the same regardless of the partition employed. As some results may depend on the system and the calculation, we encourage comparing these three partitions to each other to find incongruences.

- example02: With the already generated ```.aoms``` and ```.molinfo``` files, we can perform a fast calculation without the single-point calculation and the same information.

- example03: If the ```.molinfo``` file has not been generated, one can still obtain the information from the ```mol``` object without requiring the single-point calculation.

- example04: This is an example of the calculation using the T1 structure for benzene, both optimized and single-point. For an unrestricted calculation, the aromaticity indicators are split into alpha-alpha and beta-beta contributions, as shown in the output.

- example05: To provide FLU references to the code, one needs to provide a dictionary, ```flurefs```, with the bond symbols the DI value corresponding to it. If the pattern already exists, it will be updated, and if it does not it will be added to the existing ones. Without the ```mol``` nor the ```.molinfo``` variables, one needs to provide the list connectivity, which is a list containing the symbols in ring connectivtiy, but only one can be given at a time.

- example06: As for the HOMA (and HOMER), the user needs to provide the dictionary ```homarefs```, which contains the bond pattern and an inner dictionary containing the ```r_opt``` and ```alpha``` parameters (see the example for a better understanding of the structure of the variable. The example provides the calculation of the HOMER values for benzene singlet, which indeed produce non-aromatic values as expected. Without the molecule information, the user needs to provide the connectivtiy of the atoms and the molecular geometry as stated in the ```mol.atom_coords()``` object.

> The program will check the topology of the AOMs to separate into singlet and triplet calculations. Thus, it will only compute HOMA for singlets and HOMER for triplets.

- example07: Here we showcase how to compute the indices individually from the ESIpy functions, although we recommend using the whole ```aromaticity()``` function as the computational time is minimal (except for the MCI in large systems).

- example08: Previous computational tools could not allow the proper calculation of the MCI in large systems due to its exponential growth. We probe that not only the calculation of a 14-membered ring is possible, but also that 10-membered rings are very fast to compute (less than 10 seconds excluding the single-point calculation).

- example09: From the ```ESIpy/utils``` repository, one can access the ```write_int``` function that writes the AOMs as an input for the ESI-3D code.



