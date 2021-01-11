import numpy as np
import pandas as pd

n_days = 1099
n_weeks = (n_days // 7) if (n_days % 7 == 0) else (n_days // 7 + 1)


def parse_visits(df, ind):
    line = df.loc[ind][0]
    line = line.strip()
    visits = [int(s) for s in line.split(' ')]
    return visits


def drop_empty_weeks(V):
    for i in V.index:
        if not np.any(V.loc[i, :]):
            V = V.drop(i)
    return V


def make_visit_matrix(visits):
    V = np.zeros(shape=(n_weeks, 7), dtype=int)
    for day in visits:
        V[(day - 1) // 7, (day - 1) % 7] = 1
    V = pd.DataFrame(V, columns=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    V.index = V.index + 1
    return V.iloc[:-1], V.iloc[-1]


def is_true_prediction(V_test, predicted_day):
    return (V_test.iloc[predicted_day] == 1) and (V_test.iloc[:predicted_day].sum() == 0)


class Predictor:

    def __init__(self, lambd=0.985, gamma=0.25):
        # precomute weights
        weeks = range(1, n_weeks + 1)
        self.w_lin = np.array([(n_weeks - i - 1) / n_weeks for i in weeks])
        self.w_lin /= self.w_lin.sum()
        self.w_exp = np.array([lambd ** i for i in weeks])
        self.w_exp /= self.w_exp.sum()
        self.w_recip = np.array([1 / i ** gamma for i in weeks])
        self.w_recip /= self.w_recip.sum()
        # initialize score variable
        self.score = 0

    def get_weights(self, choose='lin'):
        if choose == 'lin':
            return self.w_lin
        elif choose == 'exp':
            return self.w_exp
        elif choose == "recip":
            return self.w_recip

    @staticmethod
    def compute_p_(V, w):
        for ind in V.index:
            V.loc[ind, :] *= w[ind]
        p = V.sum(axis=0)
        p_ = np.zeros(7)
        for i in range(7):
            p_[i] = (1 - p[:i]).prod() * p[i]
        return p_

    def fit_and_predict(self, df):
        self.predictions = pd.Series(index=df.index)
        self.score = 0
        for ind in df.index:
            visits = parse_visits(df, ind)
            V, V_test = make_visit_matrix(visits)
            V = drop_empty_weeks(V)

            predicted_day = self.predict(V)
            self.predictions.loc[ind] = predicted_day
            self.score += is_true_prediction(V_test, predicted_day)
        self.score /= df.shape[0]

    def get_score(self):
        return self.score

    def predict(self, V):
        pass
        # w = self.get_weights()
        # p_ = self.compute_p_(V, w)
        # predicted_day = np.argmax(p_)
        # return predicted_day


class WeightedProbPredictor(Predictor):
    def predict(self, V):
        w = self.get_weights()
        p_ = self.compute_p_(V, w)
        predicted_day = np.argmax(p_)
        return predicted_day


class RandomPredictor(Predictor):
    random_state = 142

    def predict(self, V):
        rng = np.random.default_rng(self.random_state)
        predicted_day = int(np.floor(rng.uniform(0, 7.99)))
        return predicted_day
