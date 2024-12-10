<p align="center"><img width="40.0%" src="https://github.com/jgrebol/ESIpy/blob/main/logoesipy.png"></p>

The ESIpy program is aimed at the calculation of population analysis and aromaticity indicators from different Hilbert-space partitions using the PySCF module. The program supports both restricted and unrestricted calculations for single-determinant wavefunctions, and correlated wavefunctions from a restricted object (RHF). The atomic partitions supported by the program are Mulliken, Löwdin, meta-Löwdin, Natural Atomic Orbitals (NAO), and Intrinsic Atomic Orbita (IAO).

## Citation

All the calculations performed for the creation and implementation of this program have been conducted in the following scientific paper:

**Joan Grèbol-Tomàs, Eduard Matito, Pedro Salvador, Chem. Eur. J. 2024, 30, e202401282.**

Also, find it on-line [here](https://chemistry-europe.onlinelibrary.wiley.com/doi/10.1002/chem.202401282?af=R). If you are publishing the results obtained from ESIpy remember to cite the program. The code is licensed under the GNU GPLv3. See the [LICENSE](LICENSE) file for details. See the [examples/README.md](examples/README.md) file for details on how to use the program. If you encounter any bugs, please feel free to report them on the [Issues page](https://github.com/jgrebol/ESIpy/issues), or send a mail to [joan.grebol@dipc.org](mailto:joan.grebol@dipc.org).

## Theoretical background

### Hilbert-space partitioning

In order to obtain information of the atomic contributions in a given chemical system (for instance atomic populations and electron sharing indices) it is crucial to define an atom in a molecule (AIM), which can either be real-space partition (allocating each point of the 3D space fully or partially to a specific atom) or Hilbert-space partition (separating the atomic basis functions belonging to a certain atom). The ESI-3D code[1] developed by Dr. Eduard Matito mainly used Bader's QTAIM[2] (real-space scheme) as the AIM for the calculations. However, in this program we propose the use of Hilbert-space schemes (Mulliken[3], Löwdin[4], Meta-Löwdin[5], NAO[6], and IAO[7]) available in the PySCF[8] framework as the partition of the system. QTAIM relies on numerical integrations, so the unavoidable errors associated to them make some of these aromaticity descriptors unviable in large systems. This newer approach, however, does not require numerical integration, but rather relies on the separation of the molecule by using their atomic basis functions, leading to an exact partition of the system. The most fundamental magnitude is the **Atomic Overlap Matrix (AOM, $\mathbf{S}^{\text{A}}$) in the Molecular Orbitals (MO, $\mathbf{\phi}$) basis**, with elements

$$S_{ij}^\text{A}=\int_{\Omega_\text{A}}\phi_i^*(\textbf{r})\phi_j(\textbf{r})\text{d}\textbf{r}.$$

The average number of electrons in a given atom can be expressed in terms of the AO basis as

$$N_{\text{A}} = \sum_{\nu\in\text{A}}^\text{M} \sum_\mu^\text{M} P_{\nu\mu}S_{\mu\nu}^\text{AO} = \sum_{\nu\in\text{A}}^\text{M} (PS^\text{AO})_{\nu\nu}$$

where we can introduce the elements of the P-matrix, $P_{\nu\mu} = 2 \sum$ $&#95;{i} ^{nocc} c_{\nu i} c_{i\mu}^+$, and the overlap matrix in the Atomic Orbitals (AO, $\chi$) basis, $S_{\mu\nu}^\text{AO}=\int\chi_\mu^{*}(\textbf{r}){\chi_\nu}(\textbf{r})d\textbf{r}$. In Mulliken's approach, one can obtain information from a specific atom by only taking into account its atomic basis functions.

Moreover, the Delocalization Index (DI, $\delta$), also referred to as Bond Order (BO)[9], measures the average number of electrons shared between two atoms A and B, as

$$\delta(\text{A,B})=\sum^\text{M}&#95;{\mu\in\text{A}}\sum^\text{M}&#95;{\nu\in\text{B}}(PS^\text{AO})&#95;{\nu\mu}(PS^\text{AO})&#95;{\mu\nu}.$$

In order to mimic the expression of the AOM as that of QTAIM, one can introduce a new auxiliary matrix, $\mathbf{\eta}^{\text{A}}$, which is a bock-truncated unit matrix with all elements being zero except $\eta&#95;{\mu\mu}^\text{A}=1$. Hence, the general expression for Mulliken's approach is the following:

$$\mathbf{S}^\text{A,Mull}=\mathbf{c}^{+}\mathbf{S}^{AO}\mathbf{\eta}^\text{A}\mathbf{c}.$$

The resulting matrix is non-symmetric due to the underlying AO basis being non-orthogonal. To overcome these issues, chemists have explored alternative Hilbert-space methods that rely on orthogonalized AO bases, mainly obtained through a unitary transformation of the original AO basis used in calculations. Löwdin first proposed the symmetric orthogonalization procedure by using $T_{\mu\nu}=S_{\mu\nu}^{-1/2}$. Following his steps, several different approaches have been reported to find more robust schemes of basis set orthogonalization, being the ones applied in this article the meta-Löwdin, Natural Atomic Orbitals (NAO). Alternatively, Knizia proposed an ingenious scheme to express in an exact number the occupied MOs of a calculation in an orthogonal basis of reduced rank, the so-called Intrinsic Atomic Orbitals (IAO) approach. In all cases, the mapping from real-space to Hilbert-space can be performed as follows:

$$\mathbf{S}^\text{A,X}=\mathbf{c}^{+}({\mathbf{T}}^{-1})^{+}\mathbf{\eta}^\text{A}\mathbf{T}^{-1}\mathbf{c}.$$

### Electron-Sharing Indices 

The ESI present in this program rely on the atomic overlap matrices. The following aromaticity indicators will be expressed in terms of the ring connectivities $\mathscr{A}=\{\text{A}_1, \text{A}_2, \cdot\cdot\cdot, \text{A}_n\}$, which represent the indices of the atoms as expressed in the `mol` object.

#### Para-delocalization index (PDI)

Fulton reported that the delocalization indices in a given aromatic 6-membered ring in the _para_ position were larger than that in the _meta_ position. From that idea, Poater and coworkers proposed to average the DIs in the para position in a 6-membered ring, so the **para-delocalization index (PDI)**[10]:

$$\text{PDI}(\mathscr{A}) = \frac{\delta&#95;{\text{A}&#95;1\text{A}&#95;4}+\delta&#95;{\text{A}&#95;2\text{A}&#95;5}+\delta&#95;{\text{A}&#95;3\text{A}&#95;6}}{3},$$

A larger PDI value indicates a more aromatic character. The index can only be calculated for rings of $n$=6, so it will not be computed for rings of different sizes.

#### I<sub>ring</sub>
Giambiagi and coworkers proposed to express an index in terms of the generalized bond order along the ring, the **I<sub>ring</sub>**[11]. That is, to account for the delocalization along the ring, following the specified connectivity:

$$\text{I}&#95;{\text{ring}}(\mathscr{A})= 2^{n} \sum_{i_1,i_2\ldots i_n} S_{i_1i_2}^{\text{A}&#95;{1}} S_{i_2i_3}^{\text{A}&#95;{2}} \cdot \cdot \cdot S_{i_ni_1}^{\text{A}&#95;{n}}$$

This index relies on the multicenter character of a molecule. A larger I<sub>ring</sub> value indicates larger aromaticity along the ring.

#### Multicenter index (MCI)

As an aim to improve the I<sub>ring</sub>, Bultinck and coworkers proposed the **Multicenter Index (MCI)**[12] by not only taking into account the Kekulé structure of the system but rather all the $n!$ possible ring connectivities generated by permuting the position of all atoms in the ring, denoted as $\mathscr{P}(\mathscr{A})$:

$$\text{MCI}(\mathscr{A}) = \frac{1}{2n} \sum_{\mathscr{P}(\mathscr{A})} \text{I}_{\text{ring}}(\mathscr{A})$$

As well as the previous indices, a larger MCI value denotes a more aromatic character. Due to the exponential growth of the calculation, we do not suggest computing the MCI for rings larger than $n$=12 for single-core processes and $n$=14 for multi-core processes. See [MCI_TIMINGS.md](MCI_TIMINGS.md) for details and timings of the algorithms.

#### AV1245 (and AVmin)

When using QTAIM as the atomic partition, the numerical integration error made the multicenter indices in large rings non-viable. Matito proposed an index that contained the multicenter character as those of I<sub>ring</sub> and MCI, but without the size-extensivity problem. Therefore, he suggested to *average all the 4c-MCI values along the ring that keep the positional relationship of 1,2,4,5*, so designing the new index AV1245[13] as follows:

$$\text{AV1245}(\mathscr{A}) = \frac{1000}{3} \sum_{i=1}^n\text{MCI}(\{\text{A}&#95;i, \text{A}&#95;{i+1}, \text{A}&#95;{i+3}, \text{A}&#95;{i+4}\})$$

where if $i>n$ $\text{A}&#95;i$ should be replaced by $\text{A}_{i-n}$. In addition, Matito defined the AVmin index as the minimum (absolute) value of all the 4-MR MCI indices that enter the AV1245 expression. A higher AV1245 value indicates more aromaticity in the system, and the index can not be computed for rings smaller than 6 centers.

#### Fluctuation Index (FLU)

The Fluctuation Index (FLU)[14] measures the resemblance of a series of tabulated $\delta$ to some typical aromatic molecules:

$$ \text{FLU}(\mathscr{A}) = \frac{1}{n} \sum_{i=1}^{n} \left[\left(\frac{V(A_i)}{V(A{i-1})} \right)^\alpha \frac{\delta(A_i, A_{i-1}) - \delta_{ref}(A_i, A_{i-1})}{\delta_{ref}(A_i, A_{i-1})} \right]^2 $$

Where one can separate it into two parts: the polarizability of the bond and the comparison to some tabulated $\delta$ (for instance, the "CC", "CN", "BN", "NN" and "CS" bonds). The index is close to zero for aromatic molecules and greater than zero in non-aromatic or antiaromatic molecules, and should not be used to study reactivity as they measure the similarity with respect to some molecule.

#### Bond Order Alternation (BOA)

The Bond Order Alternation (BOA) reflects the alternation of the delocalization indices along a conjugated circuit and is built upon the BLA premise (see below in the Geometrical Aromaticity Indicators section):

$$\text{BOA}(\mathscr{A}) = \frac{1}{n_1} \sum_{i=1}^{n_1} \delta(A_{2i-1},A_{2i}) - \frac{1}{n_2} \sum_{i=1}^{n_2} \delta(A_{2i},A_{2i+1})$$

where $n_1 = \lfloor (n+1)/2 \rfloor$ and $n_2 = \lfloor n/2 \rfloor$, being $\lfloor x \rfloor$ the floor function of $x$ returning the largest integer less or equal than $x$. As well as for the BLA index, for odd-centered closed circuits this index may fail, so instead the $\text{BOA}&#95;c$ index should be used as the comparison of $\delta(A_i, A_{i+1}) - \delta(A_{i+1}, A_{i+2})$:

$$\text{BOA}&#95;c(\mathscr{A}) = \frac{1}{N} \sum_{i=1}^{N} \left| \delta(A_{i},A_{i+1}) - \delta(A_{i+1},A_{i+2}) \right|$$

### Geometrical Aromaticity Indicators

#### HOMA and HOMER

The Harmonic Oscillator Model of Aromaticity (HOMA)[15] was defined by Kruszewski and Krygowski and relies only on geometrical data. 

$$\text{HOMA}(\mathscr{A}) = 1 - \frac{\alpha}{n} \cdot \sum_i^n (R_{opt} - R_{A_i,A_{i+1}})^2 = 1 - \frac{\alpha}{n} \cdot ((R_{opt} - \bar{R})^2 + \sum_i^n (R_{A_i,A_{i+1}} - \bar{R})^2) = 1 - (EN + GEO)$$

The formula depends on a series of tabulated $R_{opt}$, as well as the normalization factor $\alpha$ for each bond to make the index 1 for benzene and 0 and negative values for non-aromatic or antiaromatic molecules, which makes it a good option for most organic molecules but fails for newer systems. The HOMA index is separated into the EN and GEO subparts, which measure the deviation of the interatomic distance into some tabulated numbers and the variance of this interatomic distance, respectively, and are close to zero for aromatic molecules. The implemented version of this index is [15]. The HOMER aromaticity index is a reparametrization of the HOMA for the T1 state.[16] Different parameters can be introduced using the `homarefs` and `homerrefs` attributes.

#### Bond-Length Alternation (BLA)

The Bond-Length Alternation (BLA)[18] index measures the average of the bond lengths of consecutive bonds in the ring

$$ \text{BLA}(\mathscr{A}) = \frac{1}{n_1} \sum_{i=1}^{n_1} r_{A_{2i-1},A_{2i}} - \frac{1}{n_2} \sum_{i=1}^{n_2} r_{A_{2i},A_{2i+1}} $$

where $n_1 = \lfloor (n+1)/2 \rfloor$ and $n_2 = \lfloor n/2 \rfloor$, being $\lfloor x \rfloor$ the floor function of $x$ returning the largest integer less or equal than $x$. This index was designed for open chains, and thus does not provide reliable results for closed circuits with and odd number of centers, so in those cases this index should be dismissed. Instead, one can use its closed-circuits counterpart, $\text{BLA}&#95;c$:

$$ \text{BLA}&#95;c(\mathscr{A}) = \frac{1}{N} \sum_{i=1}^{N} \vert r_{A_{i},A_{i+1}} - r_{A_{i+1},A_{i+2}} \vert $$

This new definition can indeed be used for closed rings, but produces numbers that even if qualitatively agree with BLA, they do not match completely.

## Features
The object esipy.ESI() needs to be initialized with the attribute `rings` and the AOMs (the variable `Smo` or, otherwise, the `mol`, `mf` and `partition` variables to construct them. See the **Variable scope** section for more information). Further customization is also available.
- `ESI.print()`: Compute population analyses, delocalization analyses, and aromaticity indicators from the AOMs (variable Smo). Only the AOMs, `Smo`, and the ring connectivities, `rings`, are required for the calculation. Any other information will complement the calculation, although it is highly recommended to provide the `mol` and `mf` objects. See **Variable scope** for more information. Multi-core processing for the MCI calculation is supported by setting `ncores` at a number of cores different than 1, albeit the speed-up is non-linear.
- `ESI.readaoms()`: To read the matrices from ESIpy or AIMAll. If read, the variable `ESI.Smo` will contain the aforementioned matrices. The path of the folder to read can be specified in `ESI.readpath`.
- `ESI.writeaoms()`: To write the matrices in the AIMAll format. The directory with the information will be located in the working directory.

## Installation
ESIpy can be installed through:
```
pip install pyesi
```
To install PySCF, following the official guidelines from [PySCF's installation guide](https://pyscf.org/install.html).:
```
pip install --prefer-binary pyscf
```
For a detailed explanation on how to run the code and how to customize it, please see the directory [examples](examples) and the [examples/README.md](examples/README.md) file.

## Attributes scope

- ```Smo```: List of matrices or string. Contains each of the AOMs. Generated from the ```ESI.aoms()``` method. The initialization of the `ESI` object will attempt generating this attribute. Can also be a string containing the name of the saved AOMs.  
- ```rings```: List (or list of lists). Contains the indices for the connected set of atoms that form the ring. Can take the form `rings=[1,2,3,4,5,6]` for a single six-membered ring, or `rings=[[1,2,3,4,5,6],[7,8,9,10,11,12]]` for two or more separate rings.
- ```mol```: From PySCF's module. Provides information about the molecule and the basis set employed for the calculation. 
- ```mf```: From PySCF's module. Provides information about the type of calculation performed. 
- ```partition```: String. Sets the type of atomic partition of the system. Options are 'mulliken', 'lowdin', 'meta_lowdin', 'nao' and 'iao', although other ways to name them are available. Please see the [`aux.format_partition()`](esipy/tools.py) function.
- ```mci```: Boolean: Sets whether the MCI is desired to be computed. By default, True if the length of the largest ring provided is less than 12.
- ```av1245```: Boolean: Sets whether the AV1245 (and AVmin) are desired to be computed. By default, True if the length of the largest ring provided is larger than 9.
- ```flurefs```: Dictionary. Contains the structure { "Bond tpye (e.g., "CC")" : DI (e.g., 1.400) }. By default, None.
- ```homarefs```: Dictionary. Contains the structure for the HOMA calculation as a dictionary using data from Ref. [15]. By default, None.
- ```homerrefs```: Dictionary. Contains the structure for the HOMER calculation "\{Bond type \{"r\_opt": distance\}, \{"alpha": alpha\} (e.g., \{"CC" : \{"r\_opt" : 1.400, "alpha" : 200.00\} \}.}. By default, None.
- ```connectivity```: List. The atomic symbols of the centers in ring connectivity: \["C", "C", "O", "C"\] for a "C-C-O-C" ring. By default, None.
- ```geom```: List. The coordinates of the molecule as provided by the `mol.atom_coords()` PySCF function. By default, None.
- ```molinfo```: Dictionary or string. Contains information about the molecule which is internally used by the program. Can also be provided by the user to avoid redoing the single-point calculation. Can also be a string with the name of the file where it is saved in binary.
- ```name```: String. Sets the name of the molecule for the generation of the ```.int``` files.
- ```ncores```: Integer. Sets the number of cores desired for the calculation of the MCI. By default, 1.
- ```saveaoms```: String. Sets the name where to save the AOMs in binary. Recommended `.aoms` extension. By default, None.
- ```savemolinfo```: String. Sets the name where to save the dictionary with the molecular information in binary. Recommended `.molinfo` extension. By default, None.
- ```readpath```: String. Path where the `\_atomicfiles` directory is located. By default, will search for the working directory.

## Methods scope
- ```print()```: Main ESIpy program. It will compute atomic populations, electron delocalization measures and electronic aromaticity indicators, based on the `Smo` and `rings` attributes. If `Smo` is not provided, it will be generated automatically from the `mol` and `mf` attributes, coming directly from PySCF.
- ```readaoms()```: It will read the AOMs from some `.int` files. Requires setting `readpath` variable. By default, the working directory.
- ```writeaoms()```: Will write the AOMs in an analogue format to that of AIMAll. These files can be read with ESI-3D (version from 15 November 2024).

## Further work
- Approximations for the MCI calculation in large systems.
- Read the AOMs (or the data required for their calculation) from other source programs and store them as ESIpy `Smo`.
- Calculation of aromaticity indicators from defined fragments.
- Split the calculation into orbital contributions.
- Algorithm to automatically find the rings inside a system.

## References
- [1] E. Matito, in ‘ESI-3D Electron Sharing Indexes Program for 3D Molecular Space Partitioning’, Girona IQC, 2006
- [2] R. F. W. Bader, Atoms in molecules: a quantum theory, Clarendon Press; Oxford University Press, Oxford [England]: New York, 1994.
- [3] R. S. Mulliken, The Journal of Chemical Physics, 1955, 23, 1833–1840.
- [4] P.-O. Löwdin, The Journal of Chemical Physics, 1950, 18, 365–375.
- [5] A. E. Reed, R. B. Weinstock and F. Weinhold, The Journal of Chemical Physics, 1985, 83, 735–746.
- [6] Q. Sun and G. K.-L. Chan, Journal of Chemical Theory and Computation, 2014, 10, 3784–3790.
- [7] G. Knizia, Journal of Chemical Theory and Computation, 2013, 9, 4834–4843.
- [8] Q. Sun et al. The Journal of Chemical Physics, 2020, 153, 024109.
- [9] I. Mayer, Chemical Physics Letters, 1983, 97, 270–274.
- [10] J. Poater et al., Chemistry–A European Journal, 2003, 9, 400–406.
- [11] M. Giambiagi, M. S. De Giambiagi and K. C. Mundim, Structural Chemistry, 1990, 1, 423–427.
- [12] P. Bultinck, R. Ponec and S. Van Damme, Journal of Physical Organic Chemistry, 2005, 18, 706–718.
- [13] E. Matito, Physical Chemistry Chemical Physics, 2016, 18, 11839–11846.
- [14] E. Matito, M.Duran, M.Solà. The Journal of Chemical Physics, 2005, 122, 014109.
- [15] J. Kruszewski and T. M. Krygowski. Tetrahedron Lett., 13(36):3839–3842, 1972.
- [16] E. M. Arpa and B. Durbeej. Physical Chemistry Chemical Physics. 2023, 25, 16763-16771.
