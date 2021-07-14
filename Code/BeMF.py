from math import exp
import random
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


def sigmoid(x):
    try:
        return 1 / (1 + exp(-x))
    except:
        # print(f"Cannot calculate the sigmoid function for x={x}. Rounding to {1 if x > 0 else 0}")
        return 1 if x > 0 else 0


def calculate_gradient(prev_value, is_one_hot: bool, dot: float, item):
    return prev_value + (1 - sigmoid(dot)) * item if is_one_hot else prev_value - sigmoid(dot) * item


def update_factor(element, gradient, learning_rate: float, regularisation: float):
    return element + learning_rate * (gradient - regularisation * element)


class BeMF:
    num_factors = num_iters = 0
    learning_rate = regularisation = 0.0
    possible_scores = []  # eg. 1,2,3,4,5
    U = [[[]]]  # User-factor matrices for each score
    V = [[[]]]  # Item-factor matrices for each score
    user_ids = []  # All the different users
    item_ids = []  # All the different items
    number_of_users = 0
    number_of_items = 0
    ratings = [[]] # The matrix for each user-item combination with the score if the user rated it and None if not
    __cached_MAE = -1 # Caches the MAE result. Reset upon .fit() calls
    predictions_matrix = [[]] # The prediction given for each user-item combination

    def __init__(self, possible_scores: [], user_item_rating_matrix: [[]], user_ids: [], item_ids: [], num_factors: int, num_iters: int, learning_rate: int, regularisation: int, seed: int, verbose=True):
        self.num_factors = num_factors
        self.num_iters = num_iters
        self.learning_rate = learning_rate
        self.regularisation = regularisation
        self.possible_scores = possible_scores
        self.user_ids = user_ids
        self.number_of_users = len(user_ids)
        self.item_ids = item_ids
        self.number_of_items = len(item_ids)
        self.ratings = user_item_rating_matrix
        random.seed(seed)

        self.U = [[[random.random() for _ in range(num_factors)] for _ in user_ids] for _ in possible_scores]
        self.V = [[[random.random() for _ in range(num_factors)] for _ in item_ids] for _ in possible_scores]

        if verbose:
            print("*BeMF model setup completed*")
            self.print_status()

    def print_status(self):
        print(f"num_factors:\t{self.num_factors}")
        print(f"num_iters:\t{self.num_iters}")
        print(f"learning_rate:\t{self.learning_rate}")
        print(f"regularisation:\t{self.regularisation}")
        print(f"possible_scores:\t{self.possible_scores}")
        print(f"user_ids:\t{len(self.user_ids)}")
        print(f"item_ids:\t{len(self.item_ids)}")
        print(f"ratings:\t({len(self.ratings)}, {len(self.ratings[0])})")
        print(f"U:\t({len(self.U)}, {len(self.U[0])}, {len(self.U[0][0])})")
        print(f"V:\t({len(self.V)}, {len(self.V[0])}, {len(self.V[0][0])})")


    def fit(self, verbose=False, make_predictions_matrix=False):
        for i in range(1, self.num_iters+1):
            self.__cached_MAE = -1
            for s in range(len(self.possible_scores)):
                score = self.possible_scores[s]
                for user_index in range(self.number_of_users):
                    self.__update_users_factors(user_index, self.U[s], self.V[s], score)
                for item_index in range(self.number_of_items):
                    self.__update_items_factors(item_index, self.U[s], self.V[s], score)
            if verbose:
                self.__print_current_iteration(i)
        if make_predictions_matrix:
            self.make_predictions_matrix()
        if verbose:
            print("Training concluded")


    def __print_current_iteration(self, i: int):
        if i == 1:
            print("Starting fitting process. Please wait.")
            return
        if (i % 10) == 0:
            print(f"\t{i} iterations - MAE: {self.evaluate_MAE()}")
            return
        print(".", end="")

    
    def evaluate_MAE(self):
        """Calculates the Mean Absolute Error (MAE) of the model. The value should get closer to 0 as the training advances.

        Returns:
            float: The result of the calculations
        """
        if self.__cached_MAE < 0:
            pred_df = pd.DataFrame(self.make_predictions_matrix()).fillna(0)
            real_df = pd.DataFrame(self.ratings).fillna(0)
            self.__cached_MAE = mean_absolute_error(real_df, pred_df)
        return self.__cached_MAE


    def make_predictions_matrix(self):
        self.predictions_matrix = np.array([np.array([self.predict(user_index, item_index, False)
                                            for item_index in range(self.number_of_items)])
                                            for user_index in range(self.number_of_users)])
        return self.predictions_matrix



    def __update_users_factors(self, user_index: int, U: [[]], V: [[]], score: int):
        gradients = [0] * self.num_factors

        for item_index in range(len(V)):
            if not self.ratings[user_index][item_index]:
                continue  # Not rated, skip
            is_one_hot = self.ratings[user_index][item_index] == score
            dot_product = np.dot(U[user_index], V[item_index])
            gradients = [calculate_gradient(gradients[k], is_one_hot, dot_product, V[item_index][k]) for k in range(self.num_factors)]
        
        U[user_index] = [update_factor(U[user_index][k], gradients[k], self.learning_rate, self.regularisation) for k in range(self.num_factors)]


    def __update_items_factors(self, item_index: int, U: [[]], V: [[]], score: int):
        gradients = [0] * self.num_factors

        for user_index in range(len(U)):
            if not self.ratings[user_index][item_index]:
                continue  # Not rated, skip
            is_one_hot = self.ratings[user_index][item_index] == score
            dot_product = np.dot(U[user_index], V[item_index])
            gradients = [calculate_gradient(gradients[k], is_one_hot, dot_product, U[user_index][k]) for k in range(self.num_factors)]

        V[item_index] = [update_factor(V[item_index][k], gradients[k], self.learning_rate, self.regularisation) for k in range(self.num_factors)]


    def get_probability(self, user_index: int, item_index: int, score_index):
        """Calculate the probability of the user rating the item with the given score_index

        Args:

            `score_index` (int): Index of the score present in `possible_scores`

        Returns:

            float: The calculated probability
        """
        if score_index >= len(self.possible_scores):
            return f"Error: index {score_index} out of range {len(self.possible_scores)}"
        dot_product = sigmoid(np.dot(self.U[score_index][user_index], self.V[score_index][item_index]))
        sum = 0.0

        for i in range(len(self.possible_scores)):
            sum += sigmoid(np.dot(self.U[i][user_index], self.V[i][item_index]))

        try:
            return dot_product/sum
        except ZeroDivisionError:
            return 0


    def predict(self, user_index: int, item_index: int, use_cached_results: bool = True):
        """
        Args:

            `use_cached_results` (bool): If False forces recalculation of values

        Returns:

            int: the score most likely to be given by the user at `user_index` to the item at `item_index`
        """
        if user_index >= len(self.U[0]):
            return f"Error: index {user_index} out of range {len(self.U[0])}"
        if item_index >= len(self.V[0]):
            return f"Error: index {item_index} out of range {len(self.V[0])}"
        if use_cached_results:
            return self.predictions_matrix[user_index][item_index]

        maximum = self.get_probability(user_index, item_index, 0)
        index = 0

        for r in range(1, len(self.possible_scores)):
            probability = self.get_probability(user_index, item_index, r)
            if (maximum < probability):
                maximum = probability
                index = r
        
        return self.possible_scores[index]


    def predict_proba(self, user_index: int, item_index: int):
        prediction = self.predict(user_index, item_index)
        return self.get_probability(user_index, item_index, self.possible_scores.index(prediction))
