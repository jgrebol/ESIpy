Python scripts
=================

ESIpy can work on the command line using in-built Python scripts. The scripts are located in the
``scripts`` directory. This way, it is not required to build an input file. They work with the already generated
AOMs, either those that come from QTAIM partitioning (`int2esi`) or those that come from ESIpy's AOMs (`aom2esi`).
The options for the script are general and can be found in section :ref:`options`.

AOM2ESI
----------------

The general format of the command is:

```bash

aom2esi <path> [options]

where <path> is the path to the .aom and .molinfo files. The options for the script are:
* ``-r`` or ``--rings``: Specify the rings. For more than one ring, separate using a colon (e.g., "-r 1 2 3, 4 5 6").
* ``-mci``: Whether to compute the MCI.
* ``-av1245``: Whether to compute the AV1245.
* ``-n`` or ``--ncores``: Number of cores for the MCI.
* ``-v`` or ``--verbose``: Verbose.
* ``-h`` or ``--help``: Show the help message and exit.

OPTIONS
----------------



