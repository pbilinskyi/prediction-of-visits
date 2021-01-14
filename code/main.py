import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from predictor import * # WeightedProbPredictor, WeightedProbPredictor2, RandomPredictor, Ensemble, FastWPP


df = pd.read_csv('/home/paul/Documents/1DataScience/competition_dunnhumbys_shopper_challenge/data/train.csv',
                  nrows=10000, index_col=0)

# randomly select 5000 customerss
from numpy.random import default_rng
rng = default_rng(142)
ind = rng.permutation(df.shape[0])[:5000]
df = df.iloc[ind]
# df = df.iloc[:5000]

def test_algorithms():
    fwpp = FastWPP()
    wpp = WeightedProbPredictor()
    wpp2 = WeightedProbPredictor2()
    random = RandomPredictor()
    ens = Ensemble()
    for alg in [fwpp, wpp, wpp2, random, ens]:
        print("Algorithm: ", alg.class_info)
        alg.fit_and_predict(df)
        score = alg.get_score()
        print("\tFINAL score : ", score)
        sizes, scores = alg.get_learning_curve()
        plt.plot(sizes, scores)
        plt.show()
    # old_visitors = [3, 22, 86, 726, 748, 956, 1367, 1675]
    # new_visitors = [87, 378, 821, 933, 1303, 1547, 1710, 1945]
    # # print(df.loc[old_visitors])
    # wpp.fit_and_predict(df.loc[old_visitors])
    # print("score on old visitors: ", wpp.get_score())
    # wpp.fit_and_predict(df.loc[new_visitors])
    # print("score on new visitors: ", wpp.get_score())

def select_weight_scheme():
    scores = []
    fwpp = FastWPP(weight_profile="exp", lambd=0)
    lambdas = np.arange(0, 1 + 0.05, 0.05)
    for l in lambdas:
        fwpp.lambd = l
        fwpp.fit_and_predict(df)
        score = fwpp.get_score()
        scores.append(score)
        print("lambda = ", l, ", score = ", score)
    print("lambdas = " , lambdas)
    print("scores = ", scores)
    plt.plot(lambdas, np.array(scores), label = "w exp")

    scores = []
    gammas = np.arange(0, 1, 0.05)
    fwpp = FastWPP(weight_profile="recip", gamma=0)
    for g in gammas:
        fwpp.gamma = g
        fwpp.fit_and_predict(df)
        score = fwpp.get_score()
        scores.append(score)
        print("gamma = ", g, ", score = ", score)
    print("gammas = ", lambdas)
    print("scores = ", scores)
    plt.plot(gammas, np.array(scores), label="recip", color="green")

    fwwp = FastWPP(weight_profile="lin")
    fwpp.fit_and_predict(df)
    score = fwpp.get_score()
    plt.axhline(score, color='red', label='lin')

    fwwp = FastWPP(weight_profile="none")
    fwpp.fit_and_predict(df)
    score = fwpp.get_score()
    plt.axhline(score, color='black', label='without weights')
    plt.legend()

    plt.show()


select_weight_scheme()

