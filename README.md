# ESIpy
The ESIpy program is aimed to the calculation of population analysis and aromaticity indicators from different Hilbert space partitions using the PySCF module. The program supports both restricted and unrestricted calculations for single-determinant wave functions. The atomic partitions supported from the program are Mulliken, Löwdin, meta-Löwdin, Natural Atomic Orbitals (NAO) and Intrinsic Atomic Orbitals (IAO). The ESIpy respository contains the esi.py main code as well as several example scripts. 

## Theoretical background

In order to obtain information of the atomic contributions in a given chemical system (for instance atomic populations and electron sharing indices) it is crucial to define an atom in a molecule (AIM). The ESI-3D code developed by Dr. Eduard Matito mainly used QTAIM as the AIM for the calculations. However, in this program we propose the usage of Hilbert-space schemes (Mulliken, Löwdin, Meta-Löwdin, NAO and IAO) available in the PySCF framework as the partition of the system. QTAIM relies on numerical integrations, so the error accumulation in large systems becomes unviable. This newer approach, however,  does not require numerical integration, but rather relies on the separation of the molecule into each of the atomic basis functions, leading to an exact partition of the system. The most fundamental magnitude is the **Atomic Overlap Matrix (AOM, $\boldsymbol{S}^{\text{A}}$) in the Molecular Orbitals (MO, $\boldsymbol{\phi}$) basis**, with elements

$$S_{ij}^\text{A}=\int_{\Omega_\text{A}}\phi_i^*(\textbf{r})\phi_j(\textbf{r})\text{d}\textbf{r}.$$

The average number of electrons in a given atom can be expressed in terms of the atomic overlap matrix as

$$N_{\text{A}} = \sum_{\nu\in\text{A}}^\text{M} \sum_\mu^\text{M} P_{\nu\mu}S_{\mu\nu}^\text{AO} = \sum_{\nu\in\text{A}}^\text{M} (PS^\text{AO})_{\nu\nu}$$

where we can introduce the elements of the P-matrix, $P_{\nu\mu} = 2\sum_{i}^{nocc} c_{\nu i} c_{i\mu}^+$, and the overlap matrix in the Atomic Orbitals (AO) basis, $S_{\mu\nu}^\text{AO}=\int\chi_\mu^{*}(\textbf{r}){\chi_\nu}(\textbf{r})d\textbf{r}$. In Mulliken's approach, one can only take into account those atomic basis functions belonging to the specific atom.

Moreover, the Delocalization Index (DI, $\delta$), which measures the average number of electrons between two atoms A and B, as

$$\delta(\text{A,B})=\sum^\text{M}&#95;{\mu\in\text{A}}\sum^\text{M}&#95;{\nu\in\text{B}}(PS^\text{AO})&#95;{\nu\mu}(PS^\text{AO})&#95;{\mu\nu}.$$

In order to mimic the expression of the AOM as that of QTAIM, one can introduce a new auxiliary matrix, $\boldsymbol{\eta}^{\text{A}}$, which is a bock-truncated unit matrix with all elements being zero except $\eta&#95;{\mu\mu}^\text{A}=1$. Hence, the general expression for Mulliken's approach is the following:

$$\boldsymbol{S}^\text{A,Mull}=\boldsymbol{c}^{+}\boldsymbol{S}^{AO}\boldsymbol{\eta}^\text{A}\boldsymbol{c}.$$

The resulting matrix is non-symmetric due to the underlying AO basis being non-orthogonal. To overcome these issues, chemists have explored alternative Hilbert-space methods that rely on orthogonalized AO bases, mainly obtained through a unitary transformation of the original AO basis used in calculations. Löwdin first proposed the symmetric orthogonalization procedure by using $U_{\mu\nu}=S_{\mu\nu}^{\frac{1}{2}}$. Following his steps, several different approaches have been reported in order to find more robust schemes of basis set orthogonalization, being the ones applied in this article the meta-Löwdin, Natural Atomic Orbitals (NAO). Alternatively, Knizia proposed an ingenious scheme to express in an exact number the occupied MOs of a calculation in an orthogonal basis of reduced rank, the so-called Intrinsic Atomic Orbitals (IAO) approach. In all cases, the mapping from real-space to Hilbert-space can be performed as following:

$$\boldsymbol{S}^\text{A,X}=\boldsymbol{c}^{+}({\boldsymbol{U}}^{-1})^{+}\boldsymbol{\eta}^\text{A}\boldsymbol{U}^{-1}\boldsymbol{c}.$$

The ESI present in this program all rely on the atomic overlap matrices. The following aromaticity indicators will be expressed in terms of the ring connectivities $\mathscr{A}=\{\text{A}_1, \text{A}_2, \cdot\cdot\cdot, \text{A}_n\}$, which represent the indices of the atoms as expressed in the `mol` object.

## Para-delocalization index (PDI)

Fulton reported that the delocalization indices in a given aromatic 6-membered ring in _para_ position was larger than that in _meta_ position. From that idea, Poater and coworkers proposed to average the DIs in para position in a 6-membered ring, so the **para-delocalization index (PDI)**:

$$\text{PDI}(\mathscr{A}) = \frac{\delta&#95;{\text{A}&#95;1\text{A}&#95;4}+\delta&#95;{\text{A}&#95;2\text{A}&#95;5}+\delta&#95;{\text{A}&#95;3\text{A}&#95;6}}{3},$$

A larger PDI value indicates more aromatic character. The index can only be calculated for rings of $n=6$, so it will not be computed for rings of different sizes.

## I<sub>ring</sub>
Giambiagi and coworkers proposed to express an index in terms of the generalized bond order in all the ring. That is, to account for the delocalization along the ring, following the specified connectivity:

![Formula](https://render.githubusercontent.com/render/math?math=I_{\text{ring}}(\mathscr{A}) = 2^{n}\sum_{i_1,i_2...i_n} S_{i_1i_2}^{\text{A\$_1$}}S_{i_2i_3}^{\text{A\$_2$}} \cdot \cdot \cdot S_{i_ni_1}^{\text{A\$_n$}})

This index relies on the multicenter character of a molecule. A larger I<sub>ring</sub> value indicates larger aromaticity along the ring.

## Multicenter index (MCI)

As an aim to improve the I<sub>ring</sub>, Bultinck and coworkers proposed to not only take into account the Kekulé structure of the system, but rather all the $n!$ possible ring connectivities generated by permuting the position of all atoms in the ring, denoted as $\mathcal{P}(\mathcal{A})$:

\[
\text{MCI}(\mathcal{A}) = \frac{1}{2n} \sum_{\mathcal{P}(\mathcal{A})} \text{I}_{\text{ring}}(\mathcal{A}).
\]

As well as the previous indices, a larger MCI value denotes more aromatic character. Due to the exponential growth of the calculation, we do not suggest computing the MCI for rings larger than $n=12$.

## AV1245 (and AVmin)

When using QTAIM as the atomic partition, the numerical integration error made the multicenter indices in large rings non-viable. Matito proposed an index that contained the multicenter character as those of I<sub>ring</sub> and MCI, but without the size-extensivity problem. Therefore, he suggested to *average all the 4c-MCI values along the ring that keep the positional relationship of 1,2,4,5*, so designing the new index AV1245 as follows:

$$\text{AV1245}(\mathcal{A}) = \frac{1000}{3} \sum_{i=1}^n\text{MCI}(\{\text{A}_i, \text{A}_{i+1}, \text{A}_{i+3}, \text{A}_{i+4}\})$$

where if $i>n$ $\text{A}_i$ should be replaced by $\text{A}_{i-n}$. In addition, Matito defined the AV~min~ index as the minimum (absolute) value of all the 4-MR MCI indices that enter the AV1245 expression. A higher AV1245 value indicates more aromaticity in the system, and the index can not be computed for rings smaller than 6 centers.

## Features
- ``make_aoms(mol, mf, calc)``: From PySCF's `mol` and `mf` objects and `calc` as a string containing the desired partition (`mulliken`, `lowdin`, `meta_lowdin`, `nao`, `iao`), generate a list of matrices containing the Atomic Overlap Matrices (AOMs).
- `aromaticity(mol, mf, Smo, ring, calc, mci, av1245, num_threads)`: Compute population analyses, delocalization analyses and aromaticity indicators from the AOMs (variable Smo). The variable `ring` is either a list or a list of lists containing the indices of the atoms for the aromaticity calculations. `mci` and `av1245` are boolean variables to compute the MCI and AV1425 indices, respectively. Multi-core processing for the MCI calculation is supported, albeit the speed-up is non-linear.
- `aromaticity_from_aoms(Smo, ring, calc, wf, mci, av1245, num_threads)`: Compute the aromaticity indicators from the AOMs previously loaded in disk (see `scripts/05-save_aoms.py` and `scripts/06-load_aoms.py`).
- Sole functions to compute each of the aromaticity indicators (Iring, MCI, AV1245 and PDI, see `scripts/08-separate_indicators.py`).

## Utilities
- `write_int(mol, mf, molname, Smo, ring, calc)`: Writes the AOMs as an input for Dr. Eduard Matito's ESI-3D code (see `scripts/07-generate_int.py`). The atomic files are stored in a self-created directory, as well as a general input for the program (`'molname'.bad`). The ring variable is not mandatory but recommended.

## Installation
To install PySCF it is recommended to create a conda environment as follows:
```
conda create --name pyscf_env python=3.9
```
and install PySCF as:
```
conda activate pyscf_env
conda install -c pyscf_env pyscf
```
To install ESIpy in your local working stations:
```
mkdir ~/ESIpy
cd ~/ESIpy
git clone https://github.com/jgrebol/ESIpy.git
```
Add to your ```.bashrc``` file:
```
export PYTHONPATH=~/ESIpy/ESIpy:$PYTHONPATH (or the directory where it is located)
```
For a more detailed installation guide, please check [PySCF's installation guide](https://pyscf.org/install.html).

# Further work
- Function: Implementation for correlated wave functions.
- Function: Aproximations for the MCI calculation in large systems.
- Utility: Compute the exact MCI for n=14 from precomputed permutations.
