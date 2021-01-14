import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from predictor import * # WeightedProbPredictor, WeightedProbPredictor2, RandomPredictor, Ensemble, FastWPP


df = pd.read_csv('/home/paul/Documents/1DataScience/competition_dunnhumbys_shopper_challenge/data/train.csv',
                 nrows=10000, index_col=0)


def test_algorithms():
    fwpp = FastWPP()
    wpp = WeightedProbPredictor()
    wpp2 = WeightedProbPredictor2()
    random = RandomPredictor()
    ens = Ensemble()
    for alg in [fwpp]:
        print("Algorithm: ", alg.class_info)
        alg.fit_and_predict(df)
        score = alg.get_score()
        print("\tFINAL score : ", score)
        sizes, scores = alg.get_learning_curve()
        plt.plot(sizes, scores)
        plt.show()
    old_visitors = [3, 22, 86, 726, 748, 956, 1367, 1675]
    new_visitors = [87, 378, 821, 933, 1303, 1547, 1710, 1945]
    # print(df.loc[old_visitors])
    wpp.fit_and_predict(df.loc[old_visitors])
    print("score on old visitors: ", wpp.get_score())
    wpp.fit_and_predict(df.loc[new_visitors])
    print("score on new visitors: ", wpp.get_score())

test_algorithms()

# lambdas = np.arange(0, 1, 0.05)
# scores = []
# for l in lambdas:
#     wpp = WeightedProbPredictor(lambd=l)
#     wpp.fit_and_predict(df.iloc[:2000])
#     scores.append(wpp.get_score())
#     print("lambda = ", l, ", score = ", scores[l])
# scores = np.array(scores)
# plt.plot(lambdas, scores, label="WPP 1")
# plt.title("Lambdas vs Score")
# plt.show()

