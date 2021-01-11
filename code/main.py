import pandas as pd
from predictor import WeightedProbPredictor, RandomPredictor

df = pd.read_csv('/home/paul/Documents/1DataScience/competition_dunnhumbys_shopper_challenge/data/train.csv',
                 nrows=100, index_col=0)
n_days = 1099  # this number is taken from the description of the task

alg = WeightedProbPredictor()
alg.fit_and_predict(df)
score = alg.get_score()
print("score (linear WeightedProbPredictor): ", score)


alg = RandomPredictor()
alg.fit_and_predict(df)
score = alg.get_score()
print("score (random): ", score)