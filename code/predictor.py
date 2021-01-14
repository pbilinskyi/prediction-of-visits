import numpy as np
import pandas as pd

n_days = 1099  # this number is taken from the description of the task
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

    def __init__(self, weight_profile='lin', lambd=0.985, gamma=0.25):
        self.weight_profile = weight_profile
        self.lambd = lambd
        self.gamma = gamma
        self.learning_curve_x = []
        self.learning_curve_y = []
        # precomute weights
        # weeks = range(1, n_weeks + 1)
        # self.w_lin = np.array([(n_weeks - i - 1) / n_weeks for i in weeks])
        # self.w_lin /= self.w_lin.sum()
        # self.w_exp = np.array([lambd ** i for i in weeks])
        # self.w_exp /= self.w_exp.sum()
        # self.w_recip = np.array([1 / i ** gamma for i in weeks])
        # self.w_recip /= self.w_recip.sum()
        # initialize score variable
        self.score = 0

    def get_weights(self, length=n_days):
        weeks = np.arange(1, length + 1)
        if self.weight_profile == 'lin':
            w_lin = np.array([(length - i - 1) / length for i in weeks])
            w_lin /= w_lin.sum()
            return w_lin
        elif self.weight_profile == 'exp':
            w_exp = np.array([self.lambd ** i for i in weeks])
            w_exp /= w_exp.sum()
            return w_exp
        elif self.weight_profile == "recip":
            w_recip = np.array([1 / i ** self.gamma for i in weeks])
            w_recip /= w_recip.sum()
            return w_recip
        else:
            return weeks / length

    @staticmethod
    def compute_p_(V, w):
        w_sum = 0
        p = np.zeros(7)

        for i in range(V.shape[0]):
            p += V.iloc[i, :] * w[i]
            w_sum += w[i]

        #    another way (change parameter length in get_weights()):
        # for ind in V.index:
        #     p += V.loc[ind, :] * w[ind - 1]
        #     w_sum += w[ind - 1]
        p /= w_sum
        # compute p_
        p_ = np.zeros(days_in_week)
        p_[0] = p[0]
        p_negative = 1
        for i in range(1, days_in_week):
            p_negative *= (1 - p[i - 1])
            p_[i] = p_negative * p[i]
        return p_

    def predict(self, V):
        pass

    def fit_and_predict(self, df):
        n_customers = df.shape[0]
        self.score = 0
        for i, ind in enumerate(df.index):
            visits = parse_visits(df, ind)
            V, V_test = make_visit_matrix(visits)
            predicted_day = self.predict(V)
            self.score += is_true_prediction(V_test, predicted_day)
            if (i + 1) % 100 == 0:
                self.learning_curve_x.append(i + 1)
                self.learning_curve_y.append(self.score / (i + 1))
                print(f'\t{i + 1}, score = {self.score / (i + 1)}')

        self.score /= n_customers
        self.learning_curve_x.append(n_customers)
        self.learning_curve_y.append(self.score)

    def get_learning_curve(self):
        return self.learning_curve_x, self.learning_curve_y

    def get_score(self):
        return self.score


class WeightedProbPredictor(Predictor):
    class_info = "Prediction as arg max of weighted probabilities"

    def predict(self, V):
        p_ = self.predict_prob(V)
        return np.argmax(p_)

    def predict_prob(self, V):
        w = self.get_weights(length=V.shape[0])
        p_ = self.compute_p_(V, w)
        return p_


class RandomPredictor(Predictor):
    class_info = "Random guessing"
    random_state = 142

    def predict(self, V):
        rng = np.random.default_rng(self.random_state)
        predicted_day = int(np.floor(rng.uniform(0, days_in_week + 0.99)))
        return predicted_day


class WeightedProbPredictor2(WeightedProbPredictor):
    class_info = 'WPP 2'

    @staticmethod
    def compute_p_(V, w):
        # delete all visits in each week, except first
        w_sum = 0
        for i in range(V.shape[0]):
            flag = True
            for j in range(V.shape[1]):
                if (V.iloc[i, j] == 1) and flag:
                    V.iloc[i] = 0
                    V.iloc[i, j] = w[i]
                    w_sum += w[i]

        p_ = V.sum(axis=0) / w_sum
        return p_


class Ensemble(WeightedProbPredictor):
    class_info = "Ensemble of WPP1 and WPP2"
    alpha = 1
    alg1 = WeightedProbPredictor()
    alg2 = WeightedProbPredictor2()

    def predict_prob(self, V):
        p1_ = self.alg1.predict_prob(V)
        p2_ = self.alg2.predict_prob(V)
        return self.alpha * p1_ + (1 - self.alpha) * p2_


class FastWPP(Predictor):

    def fit_and_predict(self, df):
        n_customers = df.shape[0]
        self.score = 0
        for i, ind in enumerate(df.index):
            visits = parse_visits(df, ind)
            true_day = (visits[0] - 1) % 7
            visits = visits[1:]
            predicted_day = self.predict(visits)
            self.score += predicted_day == true_day
            if (i + 1) % 100 == 0:
                self.learning_curve_x.append((i+1))
                self.learning_curve_y.append(self.score / (i+1))
                print(f'\t{i + 1}, score = {self.score / (i + 1)}')
        self.score /= n_customers
        self.learning_curve_x.append(n_customers)
        self.learning_curve_y.append(self.score)

    def predict(self, visits):
        p_ = self.predict_prob(visits)
        return np.argmax(p_)

    @staticmethod
    def predict_prob(visits):
        from copy import copy
        # find number of different weeks
        d_cur = visits[0]
        num_of_week = {d_cur: 1}
        n_weeks = 1
        for d in visits[1:]:
            # if d and d_cur are not a days of the same week
            if not ((d - d_cur) < 7) and (d - 1 % 7) - (d_cur - 1 % 7) > 0:
                n_weeks += 1
                d_cur = copy(d)
            num_of_week[d] = n_weeks

        # compute weights
        weeks = range(1, n_weeks + 1)
        w = np.array([(n_weeks - i + 1) / n_weeks for i in weeks])
        w = w / w.sum()

        # compute p
        p = np.zeros(days_in_week)
        for day in visits:
            day_in_week = (day-1)%days_in_week + 1
            p[day_in_week - 1] += w[num_of_week[day] - 1]

        # compute p_
        p_ = np.zeros(days_in_week)
        p_[0] = p[0]
        p_negative = 1
        for i in range(1, days_in_week):
            p_negative *= (1 - p[i - 1])
            p_[i] = p_negative * p[i]
        return p_
