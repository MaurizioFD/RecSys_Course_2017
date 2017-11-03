#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 16/09/2017

@author: Maurizio Ferrari Dacrema
"""

import numpy as np


class Similarity_Matrix_Recommender(object):

    def __init__(self):
        super(Similarity_Matrix_Recommender, self).__init__()



    def recommend(self, user_id, n=None, exclude_seen=True, filterTopPop = False, filterCustomItems = False):

        # compute the scores using the dot product
        if self.sparse_weights:
            user_profile = self.URM_train[user_id]

            scores = user_profile.dot(self.W_sparse).toarray().ravel()

        else:

            user_profile = self.URM_train.indices[self.URM_train.indptr[user_id]:self.URM_train.indptr[user_id + 1]]
            user_ratings = self.URM_train.data[self.URM_train.indptr[user_id]:self.URM_train.indptr[user_id + 1]]

            relevant_weights = self.W[user_profile]
            scores = relevant_weights.T.dot(user_ratings)

        if self.normalize:
            # normalization will keep the scores in the same range
            # of value of the ratings in dataset
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

