#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Maurizio Ferrari Dacrema
"""

import multiprocessing
import time

import numpy as np

from Base.metrics import roc_auc, precision, recall, map, ndcg, rr
#from Base.Cython.metrics import roc_auc, precision, recall, map, ndcg, rr
from Base.Recommender_utils import check_matrix, areURMequals, removeTopPop


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


    def fit(self):
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
                                mode='sequential', filterTopPop = False,
                                filterCustomItems = np.array([], dtype=np.int),
                                filterCustomUsers = np.array([], dtype=np.int)):
        """
        Speed info:
        - Sparse weighgs: batch mode is 2x faster than sequential
        - Dense weighgts: batch and sequential speed are equivalent


        :param URM_test_new:            URM to be used for testing
        :param at: 5                    Length of the recommended items
        :param minRatingsPerUser: 1     Users with less than this number of interactions will not be evaluated
        :param exclude_seen: True       Whether to remove already seen items from the recommended items

        :param mode: 'sequential', 'parallel', 'batch'
        :param filterTopPop: False or decimal number        Percentage of items to be removed from recommended list and testing interactions
        :param filterCustomItems: Array, default empty           Items ID to NOT take into account when recommending
        :param filterCustomUsers: Array, default empty           Users ID to NOT take into account when recommending
        :return:
        """

        if len(filterCustomItems) == 0:
            self.filterCustomItems = False
        else:
            self.filterCustomItems = True
            self.filterCustomItems_ItemsID = np.array(filterCustomItems)


        if filterTopPop != False:

            self.filterTopPop = True

            _,_, self.filterTopPop_ItemsID = removeTopPop(self.URM_train, URM_2 = URM_test_new, percentageToRemove=filterTopPop)

            print("Filtering {}% TopPop items, count is: {}".format(filterTopPop*100, len(self.filterTopPop_ItemsID)))

            # Zero-out the items in order to be considered irrelevant
            URM_test_new = check_matrix(URM_test_new, format='lil')
            URM_test_new[:,self.filterTopPop_ItemsID] = 0
            URM_test_new = check_matrix(URM_test_new, format='csr')


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

        if len(filterCustomUsers) != 0:
            print("Filtering {} Users".format(len(filterCustomUsers)))
            usersToEvaluate = set(usersToEvaluate) - set(filterCustomUsers)

        usersToEvaluate = list(usersToEvaluate)



        if mode=='sequential':
            return self.evaluateRecommendationsSequential(usersToEvaluate)
        elif mode=='parallel':
            return self.evaluateRecommendationsParallel(usersToEvaluate)
        elif mode=='batch':
            return self.evaluateRecommendationsBatch(usersToEvaluate)
        # elif mode=='cython':
        #     return self.evaluateRecommendationsCython(usersToEvaluate)
        # elif mode=='random-equivalent':
        #     return self.evaluateRecommendationsRandomEquivalent(usersToEvaluate)
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



            if n_eval % 10000 == 0 or n_eval==len(usersToEvaluate)-1:
                print("Processed {} ( {:.2f}% ) in {:.2f} seconds. Users per second: {:.0f}".format(
                                  n_eval,
                                  100.0* float(n_eval+1)/len(usersToEvaluate),
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




    def evaluateRecommendationsBatch(self, usersToEvaluate, batch_size = 1000):

        roc_auc_, precision_, recall_, map_, mrr_, ndcg_ = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        n_eval = 0

        start_time = time.time()
        start_time_batch = time.time()

        #Number of blocks is rounded to the next integer
        totalNumberOfBatch = int(len(usersToEvaluate) / batch_size) + 1

        for current_batch in range(totalNumberOfBatch):

            user_first_id = current_batch*batch_size
            user_last_id = min((current_batch+1)*batch_size-1,  len(usersToEvaluate)-1)

            users_in_batch = usersToEvaluate[user_first_id:user_last_id]

            relevant_items_batch = self.URM_test[users_in_batch]

            recommended_items_batch = self.recommendBatch(users_in_batch,
                                                          exclude_seen=self.exclude_seen,
                                                          n=self.at, filterTopPop=self.filterTopPop,
                                                          filterCustomItems=self.filterCustomItems)


            for test_user in range(recommended_items_batch.shape[0]):

                n_eval += 1

                current_user = relevant_items_batch[test_user,:]

                relevant_items = current_user.indices
                recommended_items = recommended_items_batch[test_user,:]

                is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

                # evaluate the recommendation list with ranking metrics ONLY
                roc_auc_ += roc_auc(is_relevant)
                precision_ += precision(is_relevant)
                recall_ += recall(is_relevant, relevant_items)
                map_ += map(is_relevant, relevant_items)
                mrr_ += rr(is_relevant)
                ndcg_ += ndcg(recommended_items, relevant_items, relevance=current_user.data, at=self.at)



            if(time.time() - start_time_batch >= 20 or current_batch == totalNumberOfBatch-1):
                print("Processed {} ( {:.2f}% ) in {:.2f} seconds. Users per second: {:.0f}".format(
                                  n_eval,
                                  100.0* float(n_eval)/len(usersToEvaluate),
                                  time.time()-start_time,
                                  float(n_eval)/(time.time()-start_time)))

                start_time_batch = time.time()


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



    def evaluateOneUser(self, test_user):

        # Being the URM CSR, the indices are the non-zero column indexes
        #relevant_items = self.URM_test_relevantItems[test_user]
        relevant_items = self.URM_test[test_user].indices

        # this will rank top n items
        recommended_items = self.recommend(user_id=test_user, exclude_seen=self.exclude_seen,
                                           n=self.at, filterTopPop=self.filterTopPop,
                                           filterCustomItems=self.filterCustomItems)

        is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

        # evaluate the recommendation list with ranking metrics ONLY
        roc_auc_ = roc_auc(is_relevant)
        precision_ = precision(is_relevant)
        recall_ = recall(is_relevant, relevant_items)
        map_ = map(is_relevant, relevant_items)
        mrr_ = rr(is_relevant)
        ndcg_ = ndcg(recommended_items, relevant_items, relevance=self.get_user_test_ratings(test_user), at=self.at)

        return roc_auc_, precision_, recall_, map_, mrr_, ndcg_



    def evaluateRecommendationsParallel(self, usersToEvaluate):

        print("Evaluation of {} users begins".format(len(usersToEvaluate)))

        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count(), maxtasksperchild=1)
        resultList = pool.map(self.evaluateOneUser, usersToEvaluate)

        # Close the pool to avoid memory leaks
        pool.close()

        n_eval = len(usersToEvaluate)
        roc_auc_, precision_, recall_, map_, mrr_, ndcg_ = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        # Looping is slightly faster then using the numpy vectorized approach, less data transformation
        for result in resultList:
            roc_auc_ += result[0]
            precision_ += result[1]
            recall_ += result[2]
            map_ += result[3]
            mrr_ += result[4]
            ndcg_ += result[5]


        if (n_eval > 0):
            roc_auc_ = roc_auc_/n_eval
            precision_ = precision_/n_eval
            recall_ = recall_/n_eval
            map_ = map_/n_eval
            mrr_ = mrr_/n_eval
            ndcg_ =  ndcg_/n_eval

        else:
            print("WARNING: No users had a sufficient number of relevant items")


        print("Evaluated {} users".format(n_eval))

        results = {}

        results["AUC"] = roc_auc_
        results["precision"] = precision_
        results["recall"] = recall_
        results["map"] = map_
        results["NDCG"] = ndcg_
        results["MRR"] = mrr_

        return (results)



