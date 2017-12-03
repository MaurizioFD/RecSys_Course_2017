# RecSys Course 2017
This is the official repository for the 2017 Recommender Systems course at Polimi.

This repo contains a Cython implementation of:
 - SLIM BPR: It uses a Cython tree-based sparse matrix, suitable for datasets whose number of items is too big for the dense similarity matrix to fit in memory;
 - MF BPR: Matrix factorization optimizing BPR
 - FunkSVD
 - AsymmetricSVD


Cython code is already compiled for Linux. To recompile the code just set the recompile_cython flag to True.
For other OS such as Windows the c-imported numpy interface might be different (e.g. return tipe long long insead of long) therefore the code could require modifications in oder to compile.

