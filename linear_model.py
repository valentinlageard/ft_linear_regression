import numpy as np


class LinearModel():
    def __init__(self):
        self._parameters = None

    # fit the model to the data using gradient descent of the mean error
    def fit(self, regressors, regressands, learning_rate=0.0001, iterations=1000):
        mean_error_history = np.zeros(iterations)
        std_parameters = np.zeros(regressors.shape[1] + 1, dtype=np.float64)
        std_regressors, mu_regressors, sigma_regressors = self.standardize(regressors)
        std_regressands, mu_regressands, sigma_regressands = self.standardize(regressands)
        # add a column of 1 to X so the first coefficients is the intercept
        std_regressors = np.insert(std_regressors, 0, np.ones(len(std_regressors)), axis=1)
        for epoch in range(iterations):
            predictions = np.dot(std_regressors, std_parameters)
            errors = predictions - std_regressands
            std_parameters = std_parameters - learning_rate * np.dot(std_regressors.T, errors) / len(regressors)
            mean_error_history[epoch] = np.mean(errors)
        self._parameters = self.unstandardize_parameters(std_parameters, mu_regressors, mu_regressands,
                                                         sigma_regressors, sigma_regressands)
        return self._parameters, mean_error_history

    # predict
    def predict(self, X):
        if self._parameters is None:
            raise ValueError("The model hasn't been fitted yet.")
        X = np.insert(X, 0, np.ones((len(X))), axis=1)
        return np.dot(X, self._parameters)

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, parameters):
        self._parameters = parameters

    @staticmethod
    def standardize(a):
        mu = np.mean(a, axis=0)
        sigma = np.std(a, axis=0)
        std_a = (a - mu) / sigma
        return std_a, mu, sigma

    @staticmethod
    def unstandardize_parameters(std_parameters, mu_regressors, mu_regressands, sigma_regressors, sigma_regressands):
        unstandardize_transform = np.zeros((len(std_parameters), len(std_parameters)))
        unstandardize_transform[0, 1:] = -(sigma_regressands / sigma_regressors) * mu_regressors
        diagonal = sigma_regressands / sigma_regressors
        diagonal = np.insert(diagonal, 0, mu_regressands)
        np.fill_diagonal(unstandardize_transform, diagonal)
        std_parameters_without_intercept = std_parameters
        std_parameters_without_intercept[0] = 1
        return np.dot(unstandardize_transform, std_parameters_without_intercept)
