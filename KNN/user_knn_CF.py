#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/10/17

@author: Maurizio Ferrari Dacrema
"""

import numpy as np

from Base.Recommender import Recommender
from Base.Recommender_utils import check_matrix
from Base.Similarity_Matrix_Recommender import Similarity_Matrix_Recommender

try:
    from Base.Cython.cosine_similarity import Cosine_Similarity
except ImportError:
    print("Unable to load Cython Cosine_Similarity, reverting to Python")
    from Base.cosine_similarity import Cosine_Similarity


class UserKNNCFRecommender(Recommender, Similarity_Matrix_Recommender):
    """ UserKNN recommender"""

    def __init__(self, URM_train, sparse_weights=True):
        super(UserKNNCFRecommender, self).__init__()

        # Not sure if CSR here is faster
        self.URM_train = check_matrix(URM_train, 'csr')

        self.dataset = None

        self.sparse_weights = sparse_weights

    def fit(self, k=50, shrink=100, similarity='cosine', normalize=True):

        self.k = k
        self.shrink = shrink

        self.similarity = Cosine_Similarity(self.URM_train.T, shrink=shrink, topK=k, normalize=normalize, mode = similarity)

        if self.sparse_weights:
            self.W_sparse = self.similarity.compute_similarity()
        else:
            self.W = self.similarity.compute_similarity()
            self.W = self.W.toarray()





    def recommend(self, user_id, n=None, exclude_seen=True, filterTopPop = False, filterCustomItems = False):

        if n==None:
            n=self.URM_train.shape[1]-1

        # compute the scores using the dot product
        if self.sparse_weights:

            scores = self.W_sparse[user_id].dot(self.URM_train).toarray().ravel()

        else:
            # Numpy dot does not recognize sparse matrices, so we must
            # invoke the dot function on the sparse one
            scores = self.URM_train.T.dot(self.W[user_id])

        if self.normalize:
            # normalization will keep the scores in the same range
            # of value of the ratings in dataset
            user_profile = self.URM_train[user_id]

            rated = user_profile.copy()
            rated.data = np.ones_like(rated.data)
            if self.sparse_weights:
                den = rated.dot(self.W_sparse).toarray().ravel()
            else:
                den = rated.dot(self.W).ravel()
            den[np.abs(den) < 1e-6] = 1.0  # to avoid NaNs
            scores /= den

        if exclude_seen:
            scores = self._filter_seen_on_scores(user_id, scores)

        if filterTopPop:
            scores = self._filter_TopPop_on_scores(scores)

        if filterCustomItems:
            scores = self._filterCustomItems_on_scores(scores)


        # rank items and mirror column to obtain a ranking in descending score
        #ranking = scores.argsort()
        #ranking = np.flip(ranking, axis=0)

        # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
        # - Partition the data to extract the set of relevant items
        # - Sort only the relevant items
        # - Get the original item index
        relevant_items_partition = (-scores).argpartition(n)[0:n]
        relevant_items_partition_sorting = np.argsort(-scores[relevant_items_partition])
        ranking = relevant_items_partition[relevant_items_partition_sorting]


        return ranking




    def recommendBatch(self, users_in_batch, n=None, exclude_seen=True, filterTopPop = False, filterCustomItems = False):

        # compute the scores using the dot product

        if self.sparse_weights:

            scores_array = self.W_sparse[users_in_batch].dot(self.URM_train)
            scores_array = scores_array.toarray()

        else:
            # Numpy dot does not recognize sparse matrices, so we must
            # invoke the dot function on the sparse one
            scores_array = self.URM_train.T.dot(self.W[users_in_batch].T)

        if self.normalize:
            raise ValueError("Not implemented")

        # To exclude seen items perform a boolean indexing and replace their score with -inf
        # Seen items will be at the bottom of the list but there is no guarantee they'll NOT be
        # recommended
        if exclude_seen:
            user_profile_batch = self.URM_train[users_in_batch]
            scores_array[user_profile_batch.nonzero()] = -np.inf

        if filterTopPop:
            scores_array[:,self.filterTopPop_ItemsID] = -np.inf

        if filterCustomItems:
            scores_array[:, self.filterCustomItems_ItemsID] = -np.inf


        # rank items and mirror column to obtain a ranking in descending score
        #ranking = (-scores_array).argsort(axis=1)
        #ranking = np.fliplr(ranking)
        #ranking = ranking[:,0:n]

        ranking = np.zeros((scores_array.shape[0],n), dtype=np.int)

        for row_index in range(scores_array.shape[0]):
            scores = scores_array[row_index]

            relevant_items_partition = (-scores).argpartition(n)[0:n]
            relevant_items_partition_sorting = np.argsort(-scores[relevant_items_partition])
            ranking[row_index] = relevant_items_partition[relevant_items_partition_sorting]


        return ranking

