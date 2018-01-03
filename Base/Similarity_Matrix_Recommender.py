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

        if n==None:
            n=self.URM_train.shape[1]-1

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




    def recommendBatch(self, users_in_batch, n=None, exclude_seen=True, filterTopPop = False, filterCustomItems = False):

        # compute the scores using the dot product
        user_profile_batch = self.URM_train[users_in_batch]

        if self.sparse_weights:
            scores_array = np.array(user_profile_batch.dot(self.W_sparse))

        else:
            scores_array = user_profile_batch.dot(self.W)

        if self.normalize:
            raise ValueError("Not implemented")

        # To exclude seen items perform a boolean indexing and replace their score with -inf
        # Seen items will be at the bottom of the list but there is no guarantee they'll NOT be
        # recommended
        if exclude_seen:
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



    def recommend_new_user(self, user_profile, n=None, exclude_seen=True):
        # compute the scores using the dot product
        if self.sparse_weights:
            assert user_profile.shape[1] == self.W_sparse.shape[0], 'The number of items does not match!'
            scores = user_profile.dot(self.W_sparse).toarray().ravel()
        else:
            assert user_profile.shape[1] == self.W.shape[0], 'The number of items does not match!'
            scores = user_profile.dot(self.W).ravel()
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
        # rank items
        ranking = scores.argsort()[::-1]

        if exclude_seen:
            seen = user_profile.indices
            unseen_mask = np.in1d(ranking, seen, assume_unique=True, invert=True)
            ranking = ranking[unseen_mask]
        return ranking[:n]


