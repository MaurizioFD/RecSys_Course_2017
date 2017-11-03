#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Maurizio Ferrari Dacrema
"""

import multiprocessing
import time

import numpy as np

from metrics import roc_auc, precision, recall, map, ndcg, rr
from Recommender_utils import check_matrix


class Recommender(object):
    """Abstract Recommender"""

    def __init__(self):
        super(Recommender, self).__init__()
        self.URM_train = None
        self.sparse_weights = True
        self.normalize = False

        self.filterTopPop = False
        self.filterTopPop_ItemsID = np.array([], dtype=np.int)

        self.filterCustomItems = False
        self.filterCustomItems_ItemsID = np.array([], dtype=np.int)


    def fit(self, URM_train):
        pass

    def _filter_TopPop_on_scores(self, scores):
        scores[self.filterTopPop_ItemsID] = -np.inf
        return scores


    def _filterCustomItems_on_scores(self, scores):
        scores[self.filterCustomItems_ItemsID] = -np.inf
        return scores


    def _filter_seen_on_scores(self, user_id, scores):

        seen = self.URM_train.indices[self.URM_train.indptr[user_id]:self.URM_train.indptr[user_id + 1]]

        scores[seen] = -np.inf
        return scores




    def evaluateRecommendations(self, URM_test_new, at=5, minRatingsPerUser=1, exclude_seen=True,
                                mode='sequential'):
        """
        Speed info:
        - Sparse weighgs: batch mode is 2x faster than sequential
        - Dense weighgts: batch and sequential speed are equivalent


        :param URM_test_new:            URM to be used for testing
        :param at: 5                    Length of the recommended items
        :param minRatingsPerUser: 1     Users with less than this number of interactions will not be evaluated
        :param exclude_seen: True       Whether to remove already seen items from the recommended items

        :param mode: 'sequential'
        :return:
        """

        # During testing CSR is faster
        self.URM_test = check_matrix(URM_test_new, format='csr')
        self.URM_train = check_matrix(self.URM_train, format='csr')
        self.at = at
        self.minRatingsPerUser = minRatingsPerUser
        self.exclude_seen = exclude_seen


        nusers = self.URM_test.shape[0]

        # Prune users with an insufficient number of ratings
        rows = self.URM_test.indptr
        numRatings = np.ediff1d(rows)
        mask = numRatings >= minRatingsPerUser
        usersToEvaluate = np.arange(nusers)[mask]

        usersToEvaluate = list(usersToEvaluate)



        if mode=='sequential':
            return self.evaluateRecommendationsSequential(usersToEvaluate)
        else:
            raise ValueError("Mode '{}' not available".format(mode))


    def get_user_relevant_items(self, user_id):

        return self.URM_test.indices[self.URM_test.indptr[user_id]:self.URM_test.indptr[user_id+1]]

    def get_user_test_ratings(self, user_id):

        return self.URM_test.data[self.URM_test.indptr[user_id]:self.URM_test.indptr[user_id+1]]


    def evaluateRecommendationsSequential(self, usersToEvaluate):

        start_time = time.time()

        roc_auc_, precision_, recall_, map_, mrr_, ndcg_ = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        n_eval = 0

        for test_user in usersToEvaluate:

            # Calling the 'evaluateOneUser' function instead of copying its code would be cleaner, but is 20% slower

            # Being the URM CSR, the indices are the non-zero column indexes
            relevant_items = self.get_user_relevant_items(test_user)

            n_eval += 1

            recommended_items = self.recommend(user_id=test_user, exclude_seen=self.exclude_seen,
                                               n=self.at, filterTopPop=self.filterTopPop, filterCustomItems=self.filterCustomItems)

            is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

            # evaluate the recommendation list with ranking metrics ONLY
            roc_auc_ += roc_auc(is_relevant)
            precision_ += precision(is_relevant)
            recall_ += recall(is_relevant, relevant_items)
            map_ += map(is_relevant, relevant_items)
            mrr_ += rr(is_relevant)
            ndcg_ += ndcg(recommended_items, relevant_items, relevance=self.get_user_test_ratings(test_user), at=self.at)



            if(n_eval % 10000 == 0):
                print("Processed {} ( {:.2f}% ) in {:.2f} seconds. Users per second: {:.0f}".format(
                                  n_eval,
                                  100.0* float(n_eval)/len(usersToEvaluate),
                                  time.time()-start_time,
                                  float(n_eval)/(time.time()-start_time)))




        if (n_eval > 0):
            roc_auc_ /= n_eval
            precision_ /= n_eval
            recall_ /= n_eval
            map_ /= n_eval
            mrr_ /= n_eval
            ndcg_ /= n_eval

        else:
            print("WARNING: No users had a sufficient number of relevant items")

        results_run = {}

        results_run["AUC"] = roc_auc_
        results_run["precision"] = precision_
        results_run["recall"] = recall_
        results_run["map"] = map_
        results_run["NDCG"] = ndcg_
        results_run["MRR"] = mrr_

        return (results_run)

