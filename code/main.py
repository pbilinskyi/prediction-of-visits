import pandas as pd
import matplotlib.pyplot as plt
from predictor import WeightedProbPredictor, WeightedProbPredictor2, RandomPredictor

df = pd.read_csv('/home/paul/Documents/1DataScience/competition_dunnhumbys_shopper_challenge/data/train.csv',
                 nrows=200, index_col=0)
n_days = 1099  # this number is taken from the description of the task


wpp = WeightedProbPredictor()
wpp2 = WeightedProbPredictor2()
random = RandomPredictor()
# alg.fit_and_predict(df)
# score = alg.get_score()
# print("FINAL score : ", score)

for alg in [wpp2]:
    print("Algorithm: ", alg.class_info)
    alg.fit_and_predict(df)
    score = alg.get_score()
    print("\tFINAL score : ", score)
    sizes, scores = alg.get_learning_curve()
    plt.plot(sizes, scores)
    plt.show()



# alg = RandomPredictor()
# alg.fit_and_predict(df)
# score = alg.get_score()
# print("score (random): ", score)