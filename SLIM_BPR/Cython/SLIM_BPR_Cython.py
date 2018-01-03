#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 07/09/17

@author: Maurizio Ferrari Dacrema
"""

from Base.Recommender_utils import similarityMatrixTopK
from SLIM_BPR.SLIM_BPR_Python import SLIM_BPR_Python
import subprocess
import os, sys
import numpy as np


class SLIM_BPR_Cython(SLIM_BPR_Python):


    def __init__(self, URM_train, positive_threshold=4, recompile_cython = False, sparse_weights = False, sgd_mode='adagrad'):


        super(SLIM_BPR_Cython, self).__init__(URM_train,
                                              positive_threshold=positive_threshold,
                                              sparse_weights = sparse_weights)


        self.sgd_mode = sgd_mode


        if recompile_cython:
            print("Compiling in Cython")
            self.runCompilationScript()
            print("Compilation Complete")



    def fit(self, epochs=30, logFile=None, URM_test=None, filterTopPop = False, minRatingsPerUser=1,
            batch_size = 1000, validate_every_N_epochs = 1, start_validation_after_N_epochs = 0,
            lambda_i = 0.0025, lambda_j = 0.00025, learning_rate = 0.05, topK = False, sgd_mode='adagrad'):


        self.eligibleUsers = []

        # Select only positive interactions
        URM_train_positive = self.URM_train.copy()

        URM_train_positive.data = URM_train_positive.data >= self.positive_threshold
        URM_train_positive.eliminate_zeros()

        for user_id in range(self.n_users):

            start_pos = URM_train_positive.indptr[user_id]
            end_pos = URM_train_positive.indptr[user_id+1]

            if len(URM_train_positive.indices[start_pos:end_pos]) > 0:
                self.eligibleUsers.append(user_id)

        self.eligibleUsers = np.array(self.eligibleUsers, dtype=np.int64)
        self.sgd_mode = sgd_mode


        # Import compiled module
        from SLIM_BPR.Cython.SLIM_BPR_Cython_Epoch import SLIM_BPR_Cython_Epoch


        self.cythonEpoch = SLIM_BPR_Cython_Epoch(self.URM_mask,
                                                 self.sparse_weights,
                                                 self.eligibleUsers,
                                                 topK=topK,
                                                 learning_rate=learning_rate,
                                                 batch_size=1,
                                                 sgd_mode = sgd_mode)


        # Cal super.fit to start training
        super(SLIM_BPR_Cython, self).fit_alreadyInitialized(epochs=epochs,
                                         logFile=logFile,
                                         URM_test=URM_test,
                                         filterTopPop=filterTopPop,
                                         minRatingsPerUser=minRatingsPerUser,
                                         batch_size=batch_size,
                                         validate_every_N_epochs=validate_every_N_epochs,
                                         start_validation_after_N_epochs=start_validation_after_N_epochs,
                                         lambda_i = lambda_i,
                                         lambda_j = lambda_j,
                                         learning_rate = learning_rate,
                                         topK = topK)




    def runCompilationScript(self):

        # Run compile script setting the working directory to ensure the compiled file are contained in the
        # appropriate subfolder and not the project root

        compiledModuleSubfolder = "/SLIM_BPR/Cython"
        #fileToCompile_list = ['Sparse_Matrix_CSR.pyx', 'SLIM_BPR_Cython_Epoch.pyx']
        fileToCompile_list = ['SLIM_BPR_Cython_Epoch.pyx']

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
        #python compileCython.py SLIM_BPR_Cython_Epoch.pyx build_ext --inplace

        # Command to generate html report
        #subprocess.call(["cython", "-a", "SLIM_BPR_Cython_Epoch.pyx"])


    def epochIteration(self):

        self.S = self.cythonEpoch.epochIteration_Cython()

        if self.sparse_weights:
            self.W_sparse = self.S
        else:
            self.W = self.S



    def writeCurrentConfig(self, currentEpoch, results_run, logFile):

        current_config = {'learn_rate': self.learning_rate,
                          'topK_similarity': self.topK,
                          'epoch': currentEpoch,
                          'sgd_mode': self.sgd_mode}

        print("Test case: {}\nResults {}\n".format(current_config, results_run))
        # print("Weights: {}\n".format(str(list(self.weights))))

        sys.stdout.flush()

        if (logFile != None):
            logFile.write("Test case: {}, Results {}\n".format(current_config, results_run))
            # logFile.write("Weights: {}\n".format(str(list(self.weights))))
            logFile.flush()
