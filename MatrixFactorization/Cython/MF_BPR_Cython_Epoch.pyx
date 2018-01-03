"""
Created on 07/09/17

@author: Maurizio Ferrari Dacrema
"""

#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: language_level=3
#cython: nonecheck=False
#cython: cdivision=True
#cython: unpack_method_calls=True
#cython: overflowcheck=False

#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

from Base.Recommender_utils import check_matrix
import numpy as np
cimport numpy as np
import time
import sys

from libc.math cimport exp, sqrt
from libc.stdlib cimport rand, RAND_MAX


cdef struct BPR_sample:
    long user
    long pos_item
    long neg_item


cdef class MF_BPR_Cython_Epoch:

    cdef int n_users
    cdef int n_items, num_factors
    cdef int numPositiveIteractions

    cdef int useAdaGrad, rmsprop

    cdef float learning_rate, user_reg, positive_reg, negative_reg

    cdef int batch_size, sparse_weights

    cdef long[:] eligibleUsers
    cdef long numEligibleUsers

    cdef int[:] seenItemsSampledUser
    cdef int numSeenItemsSampledUser

    cdef int[:] URM_mask_indices, URM_mask_indptr

    cdef double[:,:] W, H


    def __init__(self, URM_mask, eligibleUsers, num_factors,
                 learning_rate = 0.05, user_reg = 0.0, positive_reg = 0.0, negative_reg = 0.0,
                 batch_size = 1, sgd_mode='sgd'):

        super(MF_BPR_Cython_Epoch, self).__init__()


        URM_mask = check_matrix(URM_mask, 'csr')

        self.numPositiveIteractions = int(URM_mask.nnz * 1)
        self.n_users = URM_mask.shape[0]
        self.n_items = URM_mask.shape[1]
        self.num_factors = num_factors

        self.URM_mask_indices = URM_mask.indices
        self.URM_mask_indptr = URM_mask.indptr

        # W and H cannot be initialized as zero, otherwise the gradient will always be zero
        self.W = np.random.random((self.n_users, self.num_factors))
        self.H = np.random.random((self.n_items, self.num_factors))



        if sgd_mode=='adagrad':
            self.useAdaGrad = True
        elif sgd_mode=='rmsprop':
            self.rmsprop = True
        elif sgd_mode=='sgd':
            pass
        else:
            raise ValueError(
                "SGD_mode not valid. Acceptable values are: 'sgd', 'adagrad', 'rmsprop'. Provided value was '{}'".format(
                    sgd_mode))



        self.learning_rate = learning_rate
        self.user_reg = user_reg
        self.positive_reg = positive_reg
        self.negative_reg = negative_reg


        if batch_size!=1:
            print("MiniBatch not implemented, reverting to default value 1")
        self.batch_size = 1

        self.eligibleUsers = eligibleUsers
        self.numEligibleUsers = len(eligibleUsers)


    # Using memoryview instead of the sparse matrix itself allows for much faster access
    cdef int[:] getSeenItems(self, long index):
        return self.URM_mask_indices[self.URM_mask_indptr[index]:self.URM_mask_indptr[index + 1]]



    def epochIteration_Cython(self):

        # Get number of available interactions
        cdef long totalNumberOfBatch = int(self.numPositiveIteractions / self.batch_size) + 1


        cdef BPR_sample sample
        cdef long u, i, j
        cdef long index, numCurrentBatch
        cdef double x_uij, sigmoid_user, sigmoid_item

        cdef int numSeenItems

        # Variables for AdaGrad and RMSprop
        cdef double [:] sgd_cache_item_factors, sgd_cache_user_factors
        cdef double cacheUpdate
        cdef float gamma

        cdef double H_i, H_j, W_u


        if self.useAdaGrad:
             sgd_cache_item_factors = np.zeros((self.n_items), dtype=float)
             sgd_cache_user_factors = np.zeros((self.n_users), dtype=float)

        # elif self.rmsprop:
        #     sgd_cache = np.zeros((self.n_items), dtype=float)
        #     gamma = 0.001


        cdef long start_time_epoch = time.time()
        cdef long start_time_batch = time.time()

        for numCurrentBatch in range(totalNumberOfBatch):

            # Uniform user sampling with replacement
            sample = self.sampleBatch_Cython()

            u = sample.user
            i = sample.pos_item
            j = sample.neg_item

            x_uij = 0.0

            for index in range(self.num_factors):

                x_uij = self.W[u,index] * (self.H[i,index] - self.H[j,index])

            # Use gradient of log(sigm(-x_uij))
            sigmoid_item = 1 / (1 + exp(x_uij))
            sigmoid_user = sigmoid_item




            if self.useAdaGrad:
                cacheUpdate = sigmoid_item ** 2

                sgd_cache_item_factors[i] += cacheUpdate
                sgd_cache_item_factors[j] += cacheUpdate
                sgd_cache_user_factors[u] += cacheUpdate

                sigmoid_item = sigmoid_item / (sqrt(sgd_cache_item_factors[i]) + 1e-8)
                sigmoid_user = sigmoid_user / (sqrt(sgd_cache_user_factors[u]) + 1e-8)

            #   INCOMPATIBLE CODE
            # elif self.rmsprop:
            #     cacheUpdate = sgd_cache[i] * gamma + (1 - gamma) * gradient ** 2
            #
            #     sgd_cache[i] += cacheUpdate
            #     sgd_cache[j] += cacheUpdate
            #
            #     gradient = gradient / (sqrt(sgd_cache[i]) + 1e-8)


            for index in range(self.num_factors):

                # Copy original value to avoid messing up the updates
                H_i = self.H[i, index]
                H_j = self.H[j, index]
                W_u = self.W[u, index]

                self.W[u, index] += self.learning_rate * (sigmoid_user * ( H_i - H_j ) - self.user_reg * W_u)
                self.H[i, index] += self.learning_rate * (sigmoid_item * ( W_u ) - self.positive_reg * H_i)
                self.H[j, index] += self.learning_rate * (sigmoid_item * (-W_u ) - self.negative_reg * H_j)



            if((numCurrentBatch%5000000==0 and not numCurrentBatch==0) or numCurrentBatch==totalNumberOfBatch-1):
                print("Processed {} ( {:.2f}% ) in {:.2f} seconds. Sample per second: {:.0f}".format(
                    numCurrentBatch*self.batch_size,
                    100.0* float(numCurrentBatch*self.batch_size)/self.numPositiveIteractions,
                    time.time() - start_time_batch,
                    float(numCurrentBatch*self.batch_size + 1) / (time.time() - start_time_epoch)))

                sys.stdout.flush()
                sys.stderr.flush()

                start_time_batch = time.time()


    def get_W(self):

        return np.array(self.W)


    def get_H(self):
        return np.array(self.H)



    cdef BPR_sample sampleBatch_Cython(self):

        cdef BPR_sample sample = BPR_sample()
        cdef long index
        cdef int negItemSelected



        # Warning: rand() returns an integer, in order to avoid integer division we force the
        # denominator to be a float
        cdef double RAND_MAX_DOUBLE = RAND_MAX


        #index = int(rand() / RAND_MAX_DOUBLE * self.numEligibleUsers)
        index = rand() % self.numEligibleUsers


        sample.user = self.eligibleUsers[index]

        self.seenItemsSampledUser = self.getSeenItems(sample.user)
        self.numSeenItemsSampledUser = len(self.seenItemsSampledUser)

        #index = int(rand() / RAND_MAX_DOUBLE * self.numSeenItemsSampledUser)
        index = rand() % self.numSeenItemsSampledUser


        sample.pos_item = self.seenItemsSampledUser[index]


        negItemSelected = False

        # It's faster to just try again then to build a mapping of the non-seen items
        # for every user
        while (not negItemSelected):
            #sample.neg_item = int(rand() / RAND_MAX_DOUBLE  * self.n_items)
            sample.neg_item = rand() % self.n_items

            index = 0
            while index < self.numSeenItemsSampledUser and self.seenItemsSampledUser[index]!=sample.neg_item:
                index+=1

            if index == self.numSeenItemsSampledUser:
                negItemSelected = True

        #print(sample)

        return sample
