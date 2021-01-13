import numpy as np
import pandas as pd

n_days = 1099
days_in_week = 7
n_weeks = (n_days // days_in_week) if (n_days % days_in_week == 0) else (n_days // days_in_week + 1)


def parse_visits(df, ind):
    line = df.loc[ind][0]
    line = line.strip()
    visits = [int(s) for s in line.split(' ')]
    return visits


# def drop_empty_weeks(V):
#     for i in V.index:
#         if not np.any(V.loc[i, :]):
#             V = V.drop(i)
#     return V


def make_visit_matrix(visits):
    '''
    :param visits: array with days of visit
    :return: V, matrix of visits
             V_test, matrix for testing
    '''
    V = np.zeros(shape=(n_weeks, days_in_week), dtype=int)
    for day in visits:
        V[(day - 1) // days_in_week, (day - 1) % days_in_week] = 1
    V = pd.DataFrame(V, columns=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    V.index = V.index + 1
    # drop empty weeks
    for i in V.index:
        if not np.any(V.loc[i, :]):
            V = V.drop(i)

    return V.iloc[1:], V.iloc[0]


def is_true_prediction(V_test, predicted_day):
    return (V_test.iloc[predicted_day] == 1) and (V_test.iloc[:predicted_day].sum() == 0)


class Predictor:
    class_info = "Interface for predictors"
    learning_curve = None

    def __init__(self, lambd=0.985, gamma=0.25):
        self.learning_curve_x = []
        self.learning_curve_y = []
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
        w_sum = 0
        for ind in V.index:
            V.loc[ind] *= w[ind - 1]
            w_sum += w[ind - 1]
        p = V.sum(axis=0) / w_sum
        p_ = np.zeros(days_in_week)
        for i in range(days_in_week):
            p_[i] = (1 - p[:i]).prod() * p[i]
        return p_

    def fit_and_predict(self, df):
        n_customers = df.shape[0]
        # self.predictions = pd.Series(index=df.index)
        self.score = 0
        for ind in df.index:
            visits = parse_visits(df, ind)
            V, V_test = make_visit_matrix(visits)
            # V = drop_empty_weeks(V)
            predicted_day = self.predict(V)
            # self.predictions.loc[ind] = predicted_day
            self.score += is_true_prediction(V_test, predicted_day)
            if ind % 100 == 0:
                self.learning_curve_x.append(ind)
                self.learning_curve_y.append(self.score / ind)
                print(f'\t{ind}, score = {self.score / ind}')

        self.score /= n_customers
        self.learning_curve_x.append(n_customers)
        self.learning_curve_y.append(self.score)

    def get_learning_curve(self):
        return self.learning_curve_x, self.learning_curve_y

    def get_score(self):
        return self.score

    def predict(self, V):
        pass
        # w = self.get_weights()
        # p_ = self.compute_p_(V, w)
        # predicted_day = np.argmax(p_)
        # return predicted_day


class WeightedProbPredictor(Predictor):
    class_info = "Prediction as arg max of weighted probabilities"

    def predict(self, V):
        w = self.get_weights()
        p_ = self.compute_p_(V, w)
        predicted_day = np.argmax(p_)
        return predicted_day


class RandomPredictor(Predictor):
    class_info = "Random guessing"
    random_state = 142

    def predict(self, V):
        rng = np.random.default_rng(self.random_state)
        predicted_day = int(np.floor(rng.uniform(0, days_in_week + 0.99)))
        return predicted_day


class WeightedProbPredictor2(WeightedProbPredictor):
    @staticmethod
    def compute_p_(V, w):
        # delete all visits in each week, except first
        for ind in V.index:
            flag = True
            w_sum = 0
            for i in V.columns:
                if (V.loc[ind, i] == 1) and flag:
                    V.loc[ind] = 0
                    V.loc[ind, i] = w[ind - 1]
                    w_sum += w[ind - 1]

        p_ = V.sum(axis=0) / w_sum
        return p_

