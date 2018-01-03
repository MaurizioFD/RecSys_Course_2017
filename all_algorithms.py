
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from SLIM_RMSE.SLIM_RMSE import SLIM_RMSE

from MatrixFactorization.Cython.MF_BPR_Cython import MF_BPR_Cython
from MatrixFactorization.MatrixFactorization_RMSE import FunkSVD

from KNN.user_knn_CF import UserKNNCFRecommender
from KNN.item_knn_CF import ItemKNNCFRecommender
from KNN.item_knn_CBF import ItemKNNCBFRecommender

from data.Movielens10MReader import Movielens10MReader


if __name__ == '__main__':


    dataReader = Movielens10MReader()

    URM_train = dataReader.get_URM_train()
    URM_validation = dataReader.get_URM_validation()
    URM_test = dataReader.get_URM_test()

    recommender_list = []
    recommender_list.append(ItemKNNCFRecommender(URM_train))
    recommender_list.append(UserKNNCFRecommender(URM_train))
    recommender_list.append(MF_BPR_Cython(URM_train))
    recommender_list.append(FunkSVD(URM_train))
    recommender_list.append(SLIM_BPR_Cython(URM_train, sparse_weights=False))
    recommender_list.append(SLIM_RMSE(URM_train))



    for recommender in recommender_list:

        print("Algorithm: {}".format(recommender.__class__))

        recommender.fit()

        results_run = recommender.evaluateRecommendations(URM_test, at=5, exclude_seen=True)
        print("Algorithm: {}, results: {}".format(recommender.__class__, results_run))

