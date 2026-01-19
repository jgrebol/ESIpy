.. |I_ring| replace:: I\ :math:`_{\text{ring}}`

Theoretical background
=======================

Hilbert-space partitioning
--------------------------

In order to obtain information of the atomic contributions in a given chemical system (for instance atomic populations
and electron sharing indices) it is crucial to define an atom in a molecule (AIM), which can either be real-space
partition (allocating each point of the 3D space fully or partially to a specific atom) or Hilbert-space partition (separating the atomic basis functions belonging to a certain atom).
The ESI-3D code :cite:`matito:2006esi` developed by Dr. Eduard Matito
mainly used Bader's Quantum Theory of Atoms in Molecules (QTAIM, real-space scheme) :cite:`bader:95qtaim` as the AIM for the calculations. However, in this program we propose
the use of Hilbert-space schemes (Mulliken :cite:`mulliken:55population`, Löwdin :cite:`lowdin:50ortho`, Meta-Löwdin :cite:`sun:14metalow`, NAO :cite:`reed:85nao`, and IAO :cite:`knizia:13iao`) available in the PySCF :cite:`sun:18wires,sun:20jcp,sun:18cms,sun:15jcc`
framework as the partition of the system. QTAIM relies on numerical integrations, so the unavoidable errors associated
to them make some of these aromaticity descriptors unviable in large systems. This newer approach, however, does not
require numerical integration, but rather relies on the separation of the molecule by using their atom-centered
functions, leading to an exact partition of the system. The following formulas will be expressed for **single-determinant,
closed-shell wavefunctions**. The most fundamental magnitude is the **Atomic Overlap Matrix (
AOM,** :math:`\mathbf{S}^{\text{A}}` **) in the Molecular Orbitals (MO,** :math:`\mathbf{\phi}` **) basis**, with elements

.. math::

   S_{ij}^\text{A}=\int_{\text{A}}\phi_i^*(\textbf{r})\phi_j(\textbf{r})\text{d}\textbf{r}.

The average number of electrons in a given atom (:math:`N_\text{A}`) can be expressed in terms of an Atomic Orbitals (AO, :math:`\chi`) basis as

.. math::

   N_{\text{A}} = \sum_{\nu\in\text{A}}^\text{M} \sum_\mu^\text{M} P_{\nu\mu}S_{\mu\nu}^\text{AO} = \sum_{\nu\in\text{A}}^\text{M} (PS^\text{AO})_{\nu\nu}

where we can introduce the elements of the overlap matrix in an AO basis, :math:`S_{\mu\nu}^\text{AO}=\int\chi_\mu^{*}(\textbf{r}){\chi_\nu}(\textbf{r})d\textbf{r}`.
The elements of the P-matrix, :math:`P_{\nu\mu} = 2 \sum_i ^{M} c_{\nu i} c_{i\mu}^+`, showcase the orbital occupancies.
In the simplest case of a single-determinant wavefunction, it s a unit matrix over all occupied for :math:`\text{M}>nocc` and 0s in the rest.
For multi-determinant wavefunctions, the diagonalization of the P-matrix in the MO representation gives the natural orbital occupancies (:math:`n_i`) and the
transformation matrix to the new basis (:math:`\Gamma`), which are not constrained to occupied orbitals anymore, by performing a unitary transformation:

.. math::

    \phi^{NO} = \Gamma^{+}C^{+}\chi^{\text{AO}}C\;\Gamma = (\Gamma^{'})^+\chi^{\text{AO}}\;\Gamma^{'}
