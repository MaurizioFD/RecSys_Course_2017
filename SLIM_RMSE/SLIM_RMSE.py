#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Massimo Quadrana
"""


import numpy as np
import scipy.sparse as sps
from Base.Recommender import Recommender
from Base.Recommender_utils import check_matrix
from sklearn.linear_model import ElasticNet

from Base.Similarity_Matrix_Recommender import Similarity_Matrix_Recommender
import time, sys

class SLIM_RMSE(Recommender, Similarity_Matrix_Recommender):
    """
    Train a Sparse Linear Methods (SLIM) item similarity model.
    NOTE: ElasticNet solver is parallel, a single intance of SLIM_RMSE will
          make use of half the cores available

    See:
        Efficient Top-N Recommendation by Linear Regression,
        M. Levy and K. Jack, LSRS workshop at RecSys 2013.

        SLIM: Sparse linear methods for top-n recommender systems,
        X. Ning and G. Karypis, ICDM 2011.
        http://glaros.dtc.umn.edu/gkhome/fetch/papers/SLIM2011icdm.pdf
    """

    def __init__(self, URM_train):

        super(SLIM_RMSE, self).__init__()

        self.URM_train = URM_train


    def __str__(self):
        return "SLIM (l1_penalty={},l2_penalty={},positive_only={})".format(
            self.l1_penalty, self.l2_penalty, self.positive_only
        )

    def fit(self, l1_penalty=0.1, l2_penalty=0.1, positive_only=True, topK = 100):

        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.positive_only = positive_only
        self.l1_ratio = self.l1_penalty / (self.l1_penalty + self.l2_penalty)
        self.topK = topK

        X = check_matrix(self.URM_train, 'csc', dtype=np.float32)

        n_items = X.shape[1]

        # initialize the ElasticNet model
        self.model = ElasticNet(alpha=1.0,
                                l1_ratio=self.l1_ratio,
                                positive=self.positive_only,
                                fit_intercept=False,
                                copy_X=False,
                                precompute=True,
                                selection='random',
                                max_iter=100,
                                tol=1e-4)

        # we'll store the W matrix into a sparse csr_matrix
        # let's initialize the vectors used by the sparse.csc_matrix constructor
        values, rows, cols = [], [], []
        start_time = time.time()
        start_time_printBatch = start_time

        # fit each item's factors sequentially (not in parallel)
        for currentItem in range(n_items):
            # get the target column
            y = X[:, currentItem].toarray()
            # set the j-th column of X to zero
            startptr = X.indptr[currentItem]
            endptr = X.indptr[currentItem + 1]
            bak = X.data[startptr: endptr].copy()
            X.data[startptr: endptr] = 0.0
            # fit one ElasticNet model per column
            self.model.fit(X, y)

            # self.model.coef_ contains the coefficient of the ElasticNet model
            # let's keep only the non-zero values
            #nnz_idx = self.model.coef_ > 0.0

            # Select topK values
            # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
            # - Partition the data to extract the set of relevant items
            # - Sort only the relevant items
            # - Get the original item index
            relevant_items_partition = (-self.model.coef_).argpartition(self.topK)[0:self.topK]
            relevant_items_partition_sorting = np.argsort(-self.model.coef_[relevant_items_partition])
            ranking = relevant_items_partition[relevant_items_partition_sorting]

            notZerosMask = self.model.coef_[ranking] > 0.0
            ranking = ranking[notZerosMask]

            values.extend(self.model.coef_[ranking])
            rows.extend(ranking)
            cols.extend([currentItem]*len(ranking))

            # finally, replace the original values of the j-th column
            X.data[startptr:endptr] = bak


            if time.time() - start_time_printBatch > 300:
                print("Processed {} ( {:.2f}% ) in {:.2f} minutes. Columns per second: {:.0f}".format(
                                  currentItem,
                                  100.0* float(currentItem)/n_items,
                                  (time.time()-start_time)/60,
                                  float(currentItem)/(time.time()-start_time)))
                sys.stdout.flush()
                sys.stderr.flush()

                start_time_printBatch = time.time()


        # generate the sparse weight matrix
        self.W_sparse = sps.csc_matrix((values, (rows, cols)), shape=(n_items, n_items), dtype=np.float32)




import multiprocessing
from multiprocessing import Pool
from functools import partial


class MultiThreadSLIM_RMSE(SLIM_RMSE, Similarity_Matrix_Recommender):

    def __init__(self, URM_train):

        super(MultiThreadSLIM_RMSE, self).__init__(URM_train)

    def __str__(self):
        return "SLIM_mt (l1_penalty={},l2_penalty={},positive_only={},workers={})".format(
            self.l1_penalty, self.l2_penalty, self.positive_only, self.workers
        )

    def _partial_fit(self, currentItem, X, topK):
        model = ElasticNet(alpha=1.0,
                                l1_ratio=self.l1_ratio,
                                positive=self.positive_only,
                                fit_intercept=False,
                                copy_X=False,
                                precompute=True,
                                selection='random',
                                max_iter=100,
                                tol=1e-4)

        # WARNING: make a copy of X to avoid race conditions on column j
        # TODO: We can probably come up with something better here.
        X_j = X.copy()
        # get the target column
        y = X_j[:, currentItem].toarray()
        # set the j-th column of X to zero
        X_j.data[X_j.indptr[currentItem]:X_j.indptr[currentItem + 1]] = 0.0
        # fit one ElasticNet model per column
        model.fit(X_j, y)
        # self.model.coef_ contains the coefficient of the ElasticNet model
        # let's keep only the non-zero values
        #nnz_idx = model.coef_ > 0.0

        relevant_items_partition = (-model.coef_).argpartition(topK)[0:topK]
        relevant_items_partition_sorting = np.argsort(-model.coef_[relevant_items_partition])
        ranking = relevant_items_partition[relevant_items_partition_sorting]

        notZerosMask = model.coef_[ranking] > 0.0
        ranking = ranking[notZerosMask]

        values = model.coef_[ranking]
        rows = ranking
        cols = [currentItem] * len(ranking)

        return values, rows, cols

    def fit(self,l1_penalty=0.1,
                 l2_penalty=0.1,
                 positive_only=True,
                 topK = 100,
                 workers=multiprocessing.cpu_count()):


        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.positive_only = positive_only
        self.l1_ratio = self.l1_penalty / (self.l1_penalty + self.l2_penalty)
        self.topK = topK

        self.workers = workers




        self.URM_train = check_matrix(self.URM_train, 'csc', dtype=np.float32)
        n_items = self.URM_train.shape[1]
        # fit item's factors in parallel
        
        #oggetto riferito alla funzione nel quale predefinisco parte dell'input
        _pfit = partial(self._partial_fit, X=self.URM_train, topK=self.topK)
        
        #creo un pool con un certo numero di processi
        pool = Pool(processes=self.workers)
        
        #avvio il pool passando la funzione (con la parte fissa dell'input) 
        #e il rimanente parametro, variabile
        res = pool.map(_pfit, np.arange(n_items))

        # res contains a vector of (values, rows, cols) tuples
        values, rows, cols = [], [], []
        for values_, rows_, cols_ in res:
            values.extend(values_)
            rows.extend(rows_)
            cols.extend(cols_)

        # generate the sparse weight matrix
        self.W_sparse = sps.csc_matrix((values, (rows, cols)), shape=(n_items, n_items), dtype=np.float32)

