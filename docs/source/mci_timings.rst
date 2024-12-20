MCI TIMINGS
===========

.. caution::
   This page is under construction, and thus the results should not be taken as a direct reference.

The ESIpy code provides an improved algorithm for the generation of permutations in the MCI calculation.
ESI-3D used a Nested Summation Symbol (NSS) to compute all the ring connectivities.
In ESIpy, two separate algorithms are used: **S** and **M**. Both algorithms only generate the (n-1)!/2 permutations of
a given n-lenghted list for non-symmetric AOMs and (n-1)! for symmetric AOMs (i.e., Mulliken).
In order to obtain the non-cyclic permutations, one approach computes the (n-1) cyclic permutations and appends the nth
term, resulting in the non-cyclic ring connectivities.
Moreover, due to the topology of the permutations, a factor of 2 is present if the first (n-1) elements of the list of
length n are reversed. Hence, the algorithms check this reversability and gets rid of the :math:`2` factor.

Both **S** and **M** perform slices of the original list using the `itertools.islice` module. The **S** algorithm
computes the Iring for each permutatation, while the **M** separates them into different chunks as iterators.
Then computes the MCI for each chunk and sums up all the MCIs. The storage of the slices as iterators allows for
multiprocessing without memory requirements, although the speed-up is non-linear.

By default, ESIpy works on a single core unless specified otherwise in the `num_threads` variable (if it is not
specified or it is equal to one). To use the other algorithm, **M**, set the variable to an integer greater than 1.
The following tests have been performed by generating and computing the MCI for 35x35 AOMs (a [10]annulene molecule) in
an Intel(R) Xeon(R) W-2123 CPU @ 3.60Hz) machine.

Table 1. Time (in seconds) for the MCI calculation of an n-membered ring in single-core processing, computed as an
average of three timings.

+----+--------------------+---------+
| n  | TIME AT 1 CORE (s) |         |
+----+--------------------+---------+
|    | S ALG.             | M ALG.  |
+----+--------------------+---------+
| 6  | 0.002              | 0.022   |
| 7  | 0.010              | 0.042   |
| 8  | 0.077              | 0.116   |
| 9  | 0.664              | 0.778   |
| 10 | 6.525              | 7.166   |
| 11 | 71.694             | 78.580  |
| 12 | 869.263            | 894.523 |
+----+--------------------+---------+

Table 2. Time (in seconds) for the MCI calculation of different n-membered rings and different number of cores, computed
as an average of three timings.

+----+-----------------------+---------+
| n  | Time at ≥ 1 cores (s) |         |
+----+-----------------------+---------+
|    | ncores                | M ALG.  |
+----+-----------------------+---------+
|    | 1                     | 7.166   |
|    | 2                     | 3.807   |
| 10 | 4                     | 2.436   |
|    | 6                     | 2.352   |
|    | 8                     | 2.052   |
|    | 1                     | 78.580  |
|    | 2                     | 4.0338  |
| 11 | 4                     | 31.871  |
|    | 6                     | 25.850  |
|    | 8                     | 21.159  |
|    | 1                     | 869.263 |
|    | 2                     | 470.362 |
| 12 | 4                     | 268.909 |
|    | 6                     | 273.089 |
|    | 8                     | 243.110 |
+----+-----------------------+---------+

Although the speedup is non-linear and falls at higher number of threads, we still recommend using 2-4 cores for larger
rings n=11-12.
The M algorithm has been tested for rings up to n=14 at 4 cores, which lasted around 21 hours for 42x42 AOMs (a [14]
annulene molecule) on an Intel(R) Xeon(R) Platinum 8260 CPU @ 2.40GHz machine.