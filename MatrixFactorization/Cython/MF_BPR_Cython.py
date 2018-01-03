#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 07/09/17

@author: Maurizio Ferrari Dacrema
"""

from Base.Recommender_utils import similarityMatrixTopK
from Base.Recommender import Recommender
import subprocess
import os, sys
import time
import numpy as np


class MF_BPR_Cython(Recommender):


    def __init__(self, URM_train, recompile_cython = False):


        super(MF_BPR_Cython, self).__init__()


        self.URM_train = URM_train
        self.n_users = URM_train.shape[0]
        self.n_items = URM_train.shape[1]
        self.normalize = False

        if recompile_cython:
            print("Compiling in Cython")
            self.runCompilationScript()
            print("Compilation Complete")



    def fit(self, epochs=30, logFile=None, URM_test=None, filterTopPop = False, filterCustomItems = np.array([], dtype=np.int), minRatingsPerUser=1,
            batch_size = 1000, validate_every_N_epochs = 1, start_validation_after_N_epochs = 0, num_factors=10,  positive_threshold=4,
            learning_rate = 0.05, sgd_mode='sgd', user_reg = 0.0, positive_reg = 0.0, negative_reg = 0.0):


        self.eligibleUsers = []
        self.num_factors = num_factors
        self.positive_threshold = positive_threshold

        # Select only positive interactions
        URM_train_positive = self.URM_train.copy()

        URM_train_positive.data = URM_train_positive.data >= self.positive_threshold
        URM_train_positive.eliminate_zeros()


        for user_id in range(self.n_users):

            start_pos = URM_train_positive.indptr[user_id]
            end_pos = URM_train_positive.indptr[user_id+1]

            numUserInteractions = len(URM_train_positive.indices[start_pos:end_pos])

            if  numUserInteractions > 0 and numUserInteractions<self.n_items:
                self.eligibleUsers.append(user_id)

        # self.eligibleUsers contains the userID having at least one positive interaction and one item non observed
        self.eligibleUsers = np.array(self.eligibleUsers, dtype=np.int64)
        self.sgd_mode = sgd_mode




        # Import compiled module
        from MatrixFactorization.Cython.MF_BPR_Cython_Epoch import MF_BPR_Cython_Epoch


        self.cythonEpoch = MF_BPR_Cython_Epoch(URM_train_positive,
                                                 self.eligibleUsers,
                                                 num_factors = self.num_factors,
                                                 learning_rate=learning_rate,
                                                 batch_size=1,
                                                 sgd_mode = sgd_mode,
                                                 user_reg=user_reg,
                                                 positive_reg=positive_reg,
                                                 negative_reg=negative_reg)


        self.batch_size = batch_size
        self.learning_rate = learning_rate


        start_time_train = time.time()

        for currentEpoch in range(epochs):

            start_time_epoch = time.time()

            if currentEpoch > 0:
                if self.batch_size>0:
                    self.epochIteration()
                else:
                    print("No batch not available")


            if (URM_test is not None) and (currentEpoch % validate_every_N_epochs == 0) and \
                            currentEpoch >= start_validation_after_N_epochs:

                print("Evaluation begins")

                self.W = self.cythonEpoch.get_W()
                self.H = self.cythonEpoch.get_H()

                results_run = self.evaluateRecommendations(URM_test, filterTopPop=filterTopPop,
                                                           minRatingsPerUser=minRatingsPerUser, filterCustomItems=filterCustomItems)

                self.writeCurrentConfig(currentEpoch, results_run, logFile)

                print("Epoch {} of {} complete in {:.2f} minutes".format(currentEpoch, epochs,
                                                                     float(time.time() - start_time_epoch) / 60))


            # Fit with no validation
            else:
                print("Epoch {} of {} complete in {:.2f} minutes".format(currentEpoch, epochs,
                                                                         float(time.time() - start_time_epoch) / 60))

        # Ensure W and H are up to date
        self.W = self.cythonEpoch.get_W()
        self.H = self.cythonEpoch.get_H()

        print("Fit completed in {:.2f} minutes".format(float(time.time() - start_time_train) / 60))

        sys.stdout.flush()




    def runCompilationScript(self):

        # Run compile script setting the working directory to ensure the compiled file are contained in the
        # appropriate subfolder and not the project root

        compiledModuleSubfolder = "/MatrixFactorization/Cython"
        fileToCompile_list = ['MF_BPR_Cython_Epoch.pyx']

        for fileToCompile in fileToCompile_list:

            command = ['python',
                       'compileCython.py',
                       fileToCompile,
                       'build_ext',
                       '--inplace'
                       ]


            output = subprocess.check_output(' '.join(command), shell=True, cwd=os.getcwd() + compiledModuleSubfolder)

            try:

                command = ['cython',
                           fileToCompile,
                           '-a'
                           ]

                output = subprocess.check_output(' '.join(command), shell=True, cwd=os.getcwd() + compiledModuleSubfolder)

            except:
                pass


        print("Compiled module saved in subfolder: {}".format(compiledModuleSubfolder))

        # Command to run compilation script
        #python compileCython.py MF_BPR_Cython_Epoch.pyx build_ext --inplace

        # Command to generate html report
        #subprocess.call(["cython", "-a", "MF_BPR_Cython_Epoch.pyx"])


    def epochIteration(self):

        self.cythonEpoch.epochIteration_Cython()




    def writeCurrentConfig(self, currentEpoch, results_run, logFile):

        current_config = {'learn_rate': self.learning_rate,
                          'num_factors': self.num_factors,
                          'batch_size': 1,
                          'epoch': currentEpoch}

        print("Test case: {}\nResults {}\n".format(current_config, results_run))

        sys.stdout.flush()

        if (logFile != None):
            logFile.write("Test case: {}, Results {}\n".format(current_config, results_run))
            logFile.flush()




    def recommendBatch(self, users_in_batch, n=None, exclude_seen=True, filterTopPop = False, filterCustomItems = False):

        # compute the scores using the dot product
        user_profile_batch = self.URM_train[users_in_batch]

        scores_array = np.dot(self.W[users_in_batch], self.H.T)

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



    def recommend(self, user_id, n=None, exclude_seen=True, filterTopPop = False, filterCustomItems = False):


        if n==None:
            n=self.URM_train.shape[1]-1

        scores_array = np.dot(self.W[user_id], self.H.T)

        if self.normalize:
            raise ValueError("Not implemented")


        if exclude_seen:
            scores = self._filter_seen_on_scores(user_id, scores_array)

        if filterTopPop:
            scores = self._filter_TopPop_on_scores(scores_array)

        if filterCustomItems:
            scores = self._filterCustomItems_on_scores(scores_array)


        # rank items and mirror column to obtain a ranking in descending score
        #ranking = scores.argsort()
        #ranking = np.flip(ranking, axis=0)

        # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
        # - Partition the data to extract the set of relevant items
        # - Sort only the relevant items
        # - Get the original item index
        relevant_items_partition = (-scores_array).argpartition(n)[0:n]
        relevant_items_partition_sorting = np.argsort(-scores_array[relevant_items_partition])
        ranking = relevant_items_partition[relevant_items_partition_sorting]


        return ranking
