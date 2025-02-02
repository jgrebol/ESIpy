Theoretical background
=======================

Hilbert-space partitioning
--------------------------

In order to obtain information of the atomic contributions in a given chemical system (for instance atomic populations
and electron sharing indices) it is crucial to define an atom in a molecule (AIM), which can either be real-space
partition (allocating each point of the 3D space fully or partially to a specific atom) or Hilbert-space partition (separating the atomic basis functions belonging to a certain atom). The ESI-3D code [1]_ developed by Dr. Eduard Matito
mainly used Bader's Quantum Theory of Atoms in Molecules (QTAIM, real-space scheme) [2]_ as the AIM for the calculations. However, in this program we propose
the use of Hilbert-space schemes (Mulliken [3]_, Löwdin [4]_, Meta-Löwdin [5]_, NAO [6]_, and IAO [7]_) available in the PySCF [8]_
framework as the partition of the system. QTAIM relies on numerical integrations, so the unavoidable errors associated
to them make some of these aromaticity descriptors unviable in large systems. This newer approach, however, does not
require numerical integration, but rather relies on the separation of the molecule by using their atom-centered
functions, leading to an exact partition of the system. The most fundamental magnitude is the **Atomic Overlap Matrix (
AOM,** :math:`\mathbf{S}^{\text{A}}` **) in the Molecular Orbitals (MO,** :math:`\mathbf{\phi}` **) basis**, with elements

.. math::

   S_{ij}^\text{A}=\int_{\Omega_\text{A}}\phi_i^*(\textbf{r})\phi_j(\textbf{r})\text{d}\textbf{r}.

The average number of electrons in a given atom (:math:`N_\text{A}`) can be expressed in terms of an Atomic Orbitals (AO, :math:`\chi`) basis as

.. math::

   N_{\text{A}} = \sum_{\nu\in\text{A}}^\text{M} \sum_\mu^\text{M} P_{\nu\mu}S_{\mu\nu}^\text{AO} = \sum_{\nu\in\text{A}}^\text{M} (PS^\text{AO})_{\nu\nu}

where we can introduce the elements of the overlap matrix in an AO basis, :math:`S_{\mu\nu}^\text{AO}=\int\chi_\mu^{*}(\textbf{r}){\chi_\nu}(\textbf{r})d\textbf{r}`.
The elements of the P-matrix, :math:`P_{\nu\mu} = 2 \sum_i ^{M} c_{\nu i} c_{i\mu}^+`, showcase the orbital occupancies.
In the simplest case of a single-determinant wavefunction, it takes 1s in the diagonal in all occupied orbitals (:math:`\text{M}=nocc`)and 0s in the rest.
For multi-determinant wavefunctions, the diagonalization of the P-matrix in the MO representation gives the natural orbital occupancies (:math:`n_i`) and the
transformation matrix to the new basis (:math:`\Gamma`), which are not constrained to occupied orbitals anymore, by performing a unitary transformation:

.. math::

    \phi^{NO} = \Gamma^{+}C^{+}\chi^{\text{AO}}C\;\Gamma = (\Gamma^{'})^+\chi^{\text{AO}}\;\Gamma^{'}

In this sense, :math:`\Gamma` is the diagonal representation of the MO basis and :math:`C` the transformation matrix from MOs into AOs.

In the simplest case of a single-determinant wavefunction, Mulliken's approach lets us obtain information from a specific atom by only taking into account its atomic basis functions.
Moreover, the Delocalization Index (DI, :math:`\delta`), also referred to as Bond Order (BO) [9]_, measures the average number
of electrons shared between two atoms A and B, as

.. math::

   \delta(\text{A,B})=\sum^\text{M}_{\mu\in\text{A}}\sum^\text{M}_{\nu\in\text{B}}(PS^\text{AO})_{\nu\mu}(PS^\text{AO})_{\mu\nu}.

In order to mimic the expression of the AOM as that of QTAIM, one can introduce a new auxiliary
matrix, :math:`\mathbf{\eta}^{\text{A}}`, which is a bock-truncated unit matrix with all elements being zero
except :math:`\eta_{\mu\mu}^\text{A}=1`. Hence, the general expression for Mulliken's approach is the following:

.. math::

   \mathbf{S}^\text{A,Mull}=\mathbf{c}^{+}\mathbf{S}^{AO}\mathbf{\eta}^\text{A}\mathbf{c}.

The resulting matrix is non-symmetric due to the underlying AO basis being non-orthogonal. To overcome these issues,
chemists have explored alternative Hilbert-space methods that rely on orthogonalized AO bases, mainly obtained through a
unitary transformation of the original AO basis used in calculations. Löwdin first proposed the symmetric
orthogonalization procedure by using :math:`T_{\mu\nu}=S_{\mu\nu}^{-1/2}`. Following his steps, several different approaches
have been reported to find more robust schemes of basis set orthogonalization, being the ones applied in this article
the meta-Löwdin and the Natural Atomic Orbitals (NAO). Alternatively, Knizia proposed an ingenious scheme to express in an
exact number the occupied MOs of a calculation in an orthogonal basis of reduced rank, the so-called Intrinsic Atomic
Orbitals (IAO) approach. In all cases, the mapping from real-space to Hilbert-space can be performed as follows:

.. math::

   \mathbf{S}^\text{A,X}=\mathbf{c}^{+}({\mathbf{T}}^{-1})^{+}\mathbf{\eta}^\text{A}\mathbf{T}^{-1}\mathbf{c}.

Electron-Sharing Indices
------------------------

The Eelectron Sharing Indices (**ESI **) present in this program rely on the atomic overlap matrices. The following aromaticity indicators will be
expressed in terms of the ring connectivities :math:`\mathscr{A}=\{\text{A}_1, \text{A}_2, \cdot\cdot\cdot, \text{A}_n\}`,
which represent the indices of the atoms as expressed in the `mol` object.

Para-delocalization index (PDI)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fulton reported that the delocalization indices in a given aromatic 6-membered ring in the _para_ position were larger
than that in the _meta_ position. From that idea, Poater and coworkers proposed to average the DIs in the para position
in a 6-membered ring, so the **Para-Delocalization Index (PDI)** [10]_ reads as:

.. math::

   \text{PDI}(\mathscr{A}) = \frac{\delta_{\text{A}_1\text{A}_4}+\delta_{\text{A}_2\text{A}_5}+\delta_{\text{A}_3\text{A}_6}}{3},

A larger PDI value indicates a more aromatic character. The index can only be calculated for rings of :math:`n=6 `, so it will
not be computed for rings of different sizes.

I\ :sub:`ring` \
~~~~~~~~~~~~~~

Giambiagi and coworkers proposed to express an index in terms of the generalized bond order along the ring, the **:math:`I :sub:`ring```** [11]_. That is, to account for the delocalization along the ring, following the specified connectivity:

.. math::

   \text{I}_{\text{ring}}(\mathscr{A})= 2^{n} \sum_{i_1,i_2\ldots i_n} S_{i_1i_2}^{\text{A}_{1}} S_{i_2i_3}^{\text{A}_{2}} \cdot \cdot \cdot S_{i_ni_1}^{\text{A}_{n}}

This index relies on the multicenter character of a molecule. A larger I\ :sub:`ring `\  value indicates larger
aromaticity along the ring.

Multicenter index (MCI)
~~~~~~~~~~~~~~~~~~~~~~~

As an aim to improve the I\ :sub:`ring `, Bultinck and coworkers proposed the **Multicenter Index (MCI)** [12]_ by not
only taking into account the Kekulé structure of the system but rather all the :math:`n!` possible ring connectivities
generated by permuting the position of all atoms in the ring. That way, the delocalization is measured throughout the system, rather than along the ring. Denoting the different permutations as :math:`\mathscr{P}(\mathscr{A})`:

.. math::

   \text{MCI}(\mathscr{A}) = \frac{1}{2n} \sum_{\mathscr{P}(\mathscr{A})} \text{I}_{\text{ring}}(\mathscr{A})

As well as the previous indices, a larger MCI value denotes a more aromatic character. Due to the exponential growth of
the calculation, we do not suggest computing the MCI for rings larger than :math:`n ` =12 for single-core processes and :math:`n ` =14
for multi-core processes. See :ref:`mci_timings` for details and timings of the algorithms.

AV1245 and AVmin
~~~~~~~~~~~~~~~~~~

When using real-space schemes as the atomic partition, their inherent numerical integration errors made the multicenter indices in large rings
non-viable. Matito proposed an index that contained the multicenter character as those of I\ :sub:`ring` and MCI, but
without the size-extensivity problem. Therefore, he suggested to *average all the 4c-MCI values along the ring that keep
the positional relationship of 1,2,4,5*, so designing the new index AV1245 [13]_ as follows:

.. math::

   \text{AV1245}(\mathscr{A}) = \frac{1000}{3} \sum_{i=1}^n\text{MCI}(\{\text{A}_i, \text{A}_{i+1}, \text{A}_{i+3}, \text{A}_{i+4}\})

where if :math:`i>n` :math:`\text{A}_i` should be replaced by :math:`\text{A}_{i-n}`. In addition, Matito defined the AVmin index as
the minimum (absolute) value of all the 4-MR MCI indices that enter the AV1245 expression. A higher AV1245 and AVmin values
indicates more aromaticity in the system, and the index can not be computed for rings smaller than 6 centers.

Fluctuation Index (FLU)
~~~~~~~~~~~~~~~~~~~~~~~

The Fluctuation Index (FLU) [14]_ measures the resemblance of a series of tabulated :math:`\delta` to some typical aromatic
molecules:

.. math::

   \text{FLU}(\mathscr{A}) = \frac{1}{n} \sum_{i=1}^{n} \left[\left(\frac{V(A_i)}{V(A_{i-1})} \right)^\alpha \frac{\delta(A_i, A_{i-1}) - \delta_{ref}(A_i, A_{i-1})}{\delta_{ref}(A_i, A_{i-1})} \right]^2

Where one can separate it into two parts: the polarizability of the bond and the comparison to some tabulated :math:`\delta` (
for instance, the "CC", "CN", "BN", "NN" and "CS" bonds). The index is close to zero for aromatic molecules and greater
than zero in non-aromatic or antiaromatic molecules, and should not be used to study reactivity as they measure the
similarity with respect to some molecule.

Bond Order Alternation (BOA)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Bond Order Alternation (BOA) reflects the alternation of the delocalization indices along a conjugated circuit and
is built upon the BLA premise (see below in the Geometrical Aromaticity Indicators section):

.. math::

   \text{BOA}(\mathscr{A}) = \frac{1}{n_1} \sum_{i=1}^{n_1} \delta(A_{2i-1},A_{2i}) - \frac{1}{n_2} \sum_{i=1}^{n_2} \delta(A_{2i},A_{2i+1})

where :math:`n_1 = \lfloor (n+1)/2 \rfloor` and :math:`n_2 = \lfloor n/2 \rfloor`, being :math:`\lfloor x \rfloor` the
floor function of :math:`x` returning the largest integer less or equal than :math`x`. As well as for the BLA index, for odd-centered closed
circuits this index may fail, so instead the :math:`\text{BOA}_c` index should be used as the comparison
of :math:`\delta(A_i, A_{i+1}) - \delta(A_{i+1}, A_{i+2})`:

.. math::

   \text{BOA}_c(\mathscr{A}) = \frac{1}{N} \sum_{i=1}^{N} \left| \delta(A_{i},A_{i+1}) - \delta(A_{i+1},A_{i+2}) \right|

Geometrical Aromaticity Indicators
----------------------------------

HOMA and HOMER
~~~~~~~~~~~~~~

The Harmonic Oscillator Model of Aromaticity (HOMA) [15]_ was defined by Kruszewski and Krygowski and relies only on
geometrical data.

.. math::

   \text{HOMA}(\mathscr{A}) = 1 - \frac{\alpha}{n} \cdot \sum_i^n (R_{opt} - R_{A_i,A_{i+1}})^2 = 1 - \frac{\alpha}{n} \cdot ((R_{opt} - \bar{R})^2 + \sum_i^n (R_{A_i,A_{i+1}} - \bar{R})^2) = 1 - (EN + GEO)

The formula depends on a series of tabulated optimal bon distances, :math:`R_{opt}`, as well as the normalization factor :math:`\alpha` for each bond to
make the index 1 for benzene and 0 and negative values for non-aromatic or antiaromatic molecules, which makes it a good
option for most organic molecules but fails for newer systems. The HOMA index is separated into the EN and GEO subparts,
which measure the deviation of the interatomic distance into some tabulated numbers and the variance of this interatomic
distance, respectively, and are close to zero for aromatic molecules. The implemented version of this index is [15]_. The
HOMER aromaticity index is a reparametrization of the HOMA for the lowest lying triplet excited state, T1, [16]_. Different parameters can be introduced
using the `homarefs` and `homerrefs` attributes.

Bond-Length Alternation (BLA)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Bond-Length Alternation (BLA) index measures the average of the bond lengths of consecutive bonds in the ring

.. math::

   \text{BLA}(\mathscr{A}) = \frac{1}{n_1} \sum_{i=1}^{n_1} r_{A_{2i-1},A_{2i}} - \frac{1}{n_2} \sum_{i=1}^{n_2} r_{A_{2i},A_{2i+1}}

where :math:`n_1 = \lfloor (n+1)/2 \rfloor` and :math:`n_2 = \lfloor n/2 \rfloor`, being :math:`\lfloor x \rfloor` the floor function
of :math:`x` returning the largest integer less or equal than :math:`x`. This index was designed for open chains, and thus does not
provide reliable results for closed circuits with and odd number of centers, so in those cases this index should be
dismissed. Instead, one can use its closed-circuits counterpart, :math:`\text{BLA}_c`:

.. math::

   \text{BLA}_c(\mathscr{A}) = \frac{1}{N} \sum_{i=1}^{N} \vert r_{A_{i},A_{i+1}} - r_{A_{i+1},A_{i+2}} \vert

This new definition can indeed be used for closed rings, but produces numbers that even if qualitatively agree with BLA,
they do not match completely.

References
----------

.. [1] E. Matito, in ‘ESI-3D Electron Sharing Indexes Program for 3D Molecular Space Partitioning’, Girona IQC, 2006
.. [2] R. F. W. Bader, Atoms in molecules: a quantum theory, Clarendon Press; Oxford University Press, Oxford [England]:
   New York, 1994.
.. [3] R. S. Mulliken, The Journal of Chemical Physics, 1955, 23, 1833–1840.
.. [4] P.-O. Löwdin, The Journal of Chemical Physics, 1950, 18, 365–375.
.. [5] A. E. Reed, R. B. Weinstock and F. Weinhold, The Journal of Chemical Physics, 1985, 83, 735–746.
.. [6] Q. Sun and G. K.-L. Chan, Journal of Chemical Theory and Computation, 2014, 10, 3784–3790.
.. [7] G. Knizia, Journal of Chemical Theory and Computation, 2013, 9, 4834–4843.
.. [8] Q. Sun et al. The Journal of Chemical Physics, 2020, 153, 024109.
.. [9] I. Mayer, Chemical Physics Letters, 1983, 97, 270–274.
.. [10] J. Poater et al., Chemistry–A European Journal, 2003, 9, 400–406.
.. [11] M. Giambiagi, M. S. De Giambiagi and K. C. Mundim, Structural Chemistry, 1990, 1, 423–427.
.. [12] P. Bultinck, R. Ponec and S. Van Damme, Journal of Physical Organic Chemistry, 2005, 18, 706–718.
.. [13] E. Matito, Physical Chemistry Chemical Physics, 2016, 18, 11839–11846.
.. [14] E. Matito, M.Duran, M.Solà. The Journal of Chemical Physics, 2005, 122, 014109.
.. [15] J. Kruszewski and T. M. Krygowski. Tetrahedron Lett., 13(36):3839–3842, 1972.
.. [16] E. M. Arpa and B. Durbeej. Physical Chemistry Chemical Physics. 2023, 25, 16763-16771.