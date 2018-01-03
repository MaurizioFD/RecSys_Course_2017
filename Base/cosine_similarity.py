#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/10/17

@author: Maurizio Ferrari Dacrema
"""

import numpy as np
import time, sys
import scipy.sparse as sps
from Base.Recommender_utils import check_matrix



class Cosine_Similarity:


    def __init__(self, dataMatrix, topK=100, shrink = 0, normalize = True,
                 mode = "cosine"):
        """
        Computes the cosine similarity on the columns of dataMatrix
        If it is computed on URM=|users|x|items|, pass the URM as is.
        If it is computed on ICM=|items|x|features|, pass the ICM transposed.
        :param dataMatrix:
        :param topK:
        :param shrink:
        :param normalize:
        :param mode:    "cosine"    computes Cosine similarity
                        "adjusted"  computes Adjusted Cosine, removing the average of the users
                        "pearson"   computes Pearson Correlation, removing the average of the items
                        "jaccard"   computes Jaccard similarity for binary interactions using Tanimoto
                        "tanimoto"  computes Tanimoto coefficient for binary interactions

        """

        super(Cosine_Similarity, self).__init__()

        self.TopK = topK
        self.shrink = shrink
        self.normalize = normalize
        self.n_columns = dataMatrix.shape[1]
        self.n_rows = dataMatrix.shape[0]

        self.dataMatrix = dataMatrix.copy()

        self.adjusted_cosine = False
        self.pearson_correlation = False
        self.tanimoto_coefficient = False

        if mode == "adjusted":
            self.adjusted_cosine = True
        elif mode == "pearson":
            self.pearson_correlation = True
        elif mode == "jaccard" or mode == "tanimoto":
            self.tanimoto_coefficient = True
            # Tanimoto has a specific kind of normalization
            self.normalize = False

        elif mode == "cosine":
            pass
        else:
            raise ValueError("Cosine_Similarity: value for paramether 'mode' not recognized."
                             " Allowed values are: 'cosine', 'pearson', 'adjusted', 'jaccard', 'tanimoto'."
                             " Passed value was '{}'".format(mode))



        if self.TopK == 0:
            self.W_dense = np.zeros((self.n_columns, self.n_columns))




    def applyAdjustedCosine(self):
        """
        Remove from every data point the average for the corresponding row
        :return:
        """

        self.dataMatrix = check_matrix(self.dataMatrix, 'csr')


        interactionsPerRow = np.diff(self.dataMatrix.indptr)

        nonzeroRows = interactionsPerRow > 0
        sumPerRow = np.asarray(self.dataMatrix.sum(axis=1)).ravel()

        rowAverage = np.zeros_like(sumPerRow)
        rowAverage[nonzeroRows] = sumPerRow[nonzeroRows] / interactionsPerRow[nonzeroRows]


        # Split in blocks to avoid duplicating the whole data structure
        start_row = 0
        end_row= 0

        blockSize = 1000


        while end_row < self.n_rows:

            end_row = min(self.n_rows, end_row + blockSize)

            self.dataMatrix.data[self.dataMatrix.indptr[start_row]:self.dataMatrix.indptr[end_row]] -= \
                np.repeat(rowAverage[start_row:end_row], interactionsPerRow[start_row:end_row])

            start_row += blockSize




    def applyPearsonCorrelation(self):
        """
        Remove from every data point the average for the corresponding column
        :return:
        """

        self.dataMatrix = check_matrix(self.dataMatrix, 'csc')


        interactionsPerCol = np.diff(self.dataMatrix.indptr)

        nonzeroCols = interactionsPerCol > 0
        sumPerCol = np.asarray(self.dataMatrix.sum(axis=0)).ravel()

        colAverage = np.zeros_like(sumPerCol)
        colAverage[nonzeroCols] = sumPerCol[nonzeroCols] / interactionsPerCol[nonzeroCols]


        # Split in blocks to avoid duplicating the whole data structure
        start_col = 0
        end_col= 0

        blockSize = 1000


        while end_col < self.n_columns:

            end_col = min(self.n_columns, end_col + blockSize)

            self.dataMatrix.data[self.dataMatrix.indptr[start_col]:self.dataMatrix.indptr[end_col]] -= \
                np.repeat(colAverage[start_col:end_col], interactionsPerCol[start_col:end_col])

            start_col += blockSize


    def useOnlyBooleanInteractions(self):

        # Split in blocks to avoid duplicating the whole data structure
        start_pos = 0
        end_pos= 0

        blockSize = 1000


        while end_pos < len(self.dataMatrix.data):

            end_pos = min(len(self.dataMatrix.data), end_pos + blockSize)

            self.dataMatrix.data[start_pos:end_pos] = np.ones(end_pos-start_pos)

            start_pos += blockSize




    def compute_similarity(self):

        values = []
        rows = []
        cols = []

        start_time = time.time()
        processedItems = 0

        if self.adjusted_cosine:
            self.applyAdjustedCosine()

        elif self.pearson_correlation:
            self.applyPearsonCorrelation()

        elif self.tanimoto_coefficient:
            self.useOnlyBooleanInteractions()


        # We explore the matrix column-wise
        self.dataMatrix = check_matrix(self.dataMatrix, 'csc')


        # Compute sum of squared values to be used in normalization
        sumOfSquared = np.array(self.dataMatrix.power(2).sum(axis=0)).ravel()

        # Tanimoto does not require the square root to be applied
        if not self.tanimoto_coefficient:
            sumOfSquared = np.sqrt(sumOfSquared)


        # Compute all similarities for each item using vectorization
        for columnIndex in range(self.n_columns):

            processedItems += 1

            if processedItems % 2000 == 0 or processedItems==self.n_columns:
                columnPerSec = processedItems / (time.time() - start_time)

                print("Similarity column {} ( {:2.0f} % ), {:.2f} column/sec, elapsed time {:.2f} min".format(
                    processedItems, processedItems / self.n_columns * 100, columnPerSec, (time.time() - start_time)/ 60))

                sys.stdout.flush()
                sys.stderr.flush()


            # All data points for a given item
            item_data = self.dataMatrix[:, columnIndex]
            item_data = item_data.toarray().squeeze()

            # Compute item similarities
            this_column_weights = self.dataMatrix.T.dot(item_data)
            this_column_weights[columnIndex] = 0.0

            # Apply normalization and shrinkage, ensure denominator != 0
            if self.normalize:
                denominator = sumOfSquared[columnIndex] * sumOfSquared + self.shrink + 1e-6
                this_column_weights = np.multiply(this_column_weights, 1 / denominator)

            # Apply the specific denominator for Tanimoto
            elif self.tanimoto_coefficient:
                denominator = sumOfSquared[columnIndex] + sumOfSquared - this_column_weights + self.shrink + 1e-6
                this_column_weights = np.multiply(this_column_weights, 1 / denominator)

            # If no normalization or tanimoto is selected, apply only shrink
            elif self.shrink != 0:
                this_column_weights = this_column_weights/self.shrink


            if self.TopK == 0:
                self.W_dense[:, columnIndex] = this_column_weights

            else:
                # Sort indices and select TopK
                # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
                # - Partition the data to extract the set of relevant items
                # - Sort only the relevant items
                # - Get the original item index
                relevant_items_partition = (-this_column_weights).argpartition(self.TopK-1)[0:self.TopK]
                relevant_items_partition_sorting = np.argsort(-this_column_weights[relevant_items_partition])
                top_k_idx = relevant_items_partition[relevant_items_partition_sorting]

                # Incrementally build sparse matrix
                values.extend(this_column_weights[top_k_idx])
                rows.extend(top_k_idx)
                cols.extend(np.ones(self.TopK) * columnIndex)

        if self.TopK == 0:
            return self.W_dense

        else:

            W_sparse = sps.csr_matrix((values, (rows, cols)),
                                      shape=(self.n_columns, self.n_columns),
                                      dtype=np.float32)


            return W_sparse