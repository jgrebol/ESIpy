# Examples

In this section we will go through the process of performing all the available calculations using ESIpy's functions based on the provided examples. The program is intended to be installed locally in computer clusters. To run the code from the terminal, generate the Python script or adapt those of this repository and run it as ```python code.py``` or ```python3 code.py```. To save the output as a file, use ```python code.py > code.esi```. Even though the ```.esi``` extension is not mandatory, we recommend using it. The test files have been generated from benzene and naphthalene at singlet (and triplet) level of theory, and are just set to showcase the capabilities of the code.

## Core ESIpy

ESipy works on the obect `ESI`, which will contain all the information required for the calculation. It is recommended to initialize the object with all the data, rather than adding it once the initialization process is finished. 

We will go through the list of examples explaining and highlighting some key notes:

- example01: The `ESI` object is initialized through the `mol` and `mf` objects coming from PySCF. The AOMs and the molecular information are automatically generated and stored in `Smo`. They can be saved by using the `saveaoms` and `savemolinfo` attributes in the initialization process. To print the output of the code, simply use the method `print()`.

> [!TIP]
> We strongly recommend using ```meta_lowdin```, ```nao``` and ```iao``` as the atomic partitions as they have shown to be highly basis-set independent and reliable. We introduce the five atomic partitions available at ESIpy in a for-loop scheme, although one partition can be introduced for each calculation. The computation time is the same regardless of the partition employed. As some results may depend on the system and the calculation, we encourage comparing these three partitions to each other to find incongruences.

> [!WARNING]
> In PySCF versions downloaded later than 23rd July 2023, there is a bug in the symmetry-average of NAO schemes (issue [#1755](https://github.com/pyscf/pyscf/issues/1755), bug fixed in [#1803](https://github.com/pyscf/pyscf/pull/1803)).

> [!WARNING]
> In PySCF, the `mol.spin` object represents the number of unpaired electrons. It is not the spin of the molecule. For instance, `mol.spin = 0` is a singlet state.

- example02: With the already generated ```.aoms``` and ```.molinfo``` files, we can perform a fast aromaticity calculation without any re-calculation, which will provide the same information as that coming directly from the single-point.

> [!NOTE]
> If the ```.molinfo``` file has not been generated, one can still obtain the information from the ```mol``` object without requiring the single-point calculation: just do not include the `mf` object.

- example03: The program will set the AOMs from an unrestricted calculation as `[Smo_alpha, Smo_beta]`. The indicators are calculated from alpha-alpha and beta-beta contributions.

- example04: To provide FLU references to the code, one needs to provide a dictionary, ```flurefs```, with the bond symbols the DI value corresponding to it. If the pattern already exists, it will be updated, and if it does not it will be added to the existing ones. Without the ```mol``` nor the ```.molinfo``` variables, one needs to provide the list connectivity, which is a list containing the symbols in ring connectivity, but only one can be given at a time.

> [!WARNING]
> The `partition` variable is mandatory for reference-based indices. The FLU, HOMA and HOMER will not be computed if no partition is specified unless the connectivity and reference has been explicitly specified.

- example05: Custom references for the HOMA can be introduced as stated in Ref. 15 of the main README.md. The custom "r_opt" and "alpha" parameters need to be given in the `homerrefs` attribute.

> [!NOTE]
> The program will check the topology of the AOMs to separate into singlet and triplet calculations. Thus, it will only compute HOMA for singlets and HOMER for triplets.

- example06: Individual calculation of the indicators can be performed. However, we strongly suggest using `ESI.print()` as the computational time is minimal for indices other than MCI for large rings.

- example07: Multi-processing is allowed for the calculation of the MCI, primarily in large systems. For more detailed information, please see [ESIpy/MCI_TIMINGS](ESIpy/MCI_TIMINGS).


## Treating .int files

- example08: The method `writeaoms()` allows to write the AOMs into a file readable for both ESIpy and ESI-3D programs. 

- example09: Equally, the AOMs from the `.int` files can be loaded into `Smo` objects, throuhg the `readaoms()` method. This function supports reading the AOMs from ESIpy and AIMAll from both restricted and unrestricted single-determinant calculations.

## Correlated wavefunctions

- example10: Aromaticity indicators use Fulton's approximation for the aromaticity indicators in correlated wavefunctions. If the `writeaoms()` is requested, it will additionally create a custom `.wfx` file with the occupation numbers as input for the ESI-3D program.
