#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/10/17

@author: Maurizio Ferrari Dacrema
"""


import numpy as np
import scipy.sparse as sps
from Base.Recommender_utils import check_matrix
from Base.Similarity_Matrix_Recommender import Similarity_Matrix_Recommender
from Base.Recommender import Recommender


class ItemKNNCustomSimilarityRecommender(Recommender, Similarity_Matrix_Recommender):
    """ ItemKNN recommender"""

    def __init__(self, k=50, shrinkage=100, normalize=False, sparse_weights=True):
        super(ItemKNNCustomSimilarityRecommender, self).__init__()
        self.k = k
        self.shrinkage = shrinkage
        self.normalize = normalize
        self.dataset = None
        self.similarity_name = None
        self.sparse_weights = sparse_weights


    def __str__(self):
        return "ItemKNNCBF(similarity={},k={},shrinkage={},normalize={},sparse_weights={})".format(
            self.similarity_name, self.k, self.shrinkage, self.normalize, self.sparse_weights)

    def fit(self, item_weights, URM_train, selectTopK = False):

        self.URM_train = check_matrix(URM_train, format='csc')

        # If no topK selection is required, just save the similarity
        if (not selectTopK):
            if isinstance(item_weights, np.ndarray):
                #self.W = item_weights
                #self.sparse_weights = False
                self.W_sparse = sps.csr_matrix(item_weights)
                self.sparse_weights = True
            else:
                self.W_sparse = check_matrix(item_weights, format='csr')
                self.sparse_weights = True

            return


        # If matrix is not dense, make it dense to select top K
        if not isinstance(item_weights, np.ndarray):
            item_weights = item_weights.toarray()


        idx_sorted = np.argsort(item_weights, axis=0)  # sort by column

        # for each column, keep only the top-k scored items

        if not self.sparse_weights:
            self.W = item_weights.copy()
            # index of the items that don't belong to the top-k similar items of each column
            not_top_k = idx_sorted[:-self.k, :]
            # use numpy fancy indexing to zero-out the values in sim without using a for loop
            self.W[not_top_k, np.arange(item_weights.shape[1])] = 0.0
        else:
            # iterate over each column and keep only the top-k similar items
            values, rows, cols = [], [], []
            nitems = self.URM_train.shape[1]
            for i in range(nitems):

                top_k_idx = idx_sorted[-self.k:, i]

                values.extend(item_weights[top_k_idx, i])
                rows.extend(np.arange(nitems)[top_k_idx])
                cols.extend(np.ones(self.k) * i)

                # During testing CSR is faster
            self.W_sparse = sps.csr_matrix((values, (rows, cols)), shape=(nitems, nitems), dtype=np.float32)

        #self.scoresAll = URM_train.dot(self.W_sparse)
