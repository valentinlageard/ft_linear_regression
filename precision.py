import os
import csv
import numpy as np
from linear_model import LinearModel
import matplotlib.pyplot as plt


def main():
    with open('data.csv') as data_file:
        data_reader = csv.reader(data_file)
        raw_data = list(data_reader)
        # Slice to remove the first row (column names), convert to float and store in a numpy array
        data = np.array([[float(row[0]), float(row[1])] for row in raw_data[1:]])
        y = data[:, 1]
        X = data[:, 0:1]

        parameters = np.zeros(2, dtype='float64')
        if os.path.exists('parameters'):
            with open('parameters') as file:
                parameters = np.array(file.read().split(', '), dtype='float64')
        
        model = LinearModel()
        model.parameters = parameters
        predictions = model.predict(X)
        sample_size = len(y)
        sum_of_squared_residuals = np.sum((y - predictions) ** 2)
        sum_of_squared_errors = np.sum((y - np.mean(predictions)) ** 2)
        r_squared = 1 - (sum_of_squared_residuals / sum_of_squared_errors)
        standardized_ys, _, _ = model.standardize(y)
        standardized_xs, _, _ = model.standardize(X)
        standardized_xs = standardized_xs.reshape(standardized_xs.shape[0])
        correlation_coefficient = np.dot(standardized_xs, standardized_ys) / sample_size
        regression_standard_error = np.sqrt(sum_of_squared_errors / sample_size - 2)
        standard_error_slope = np.sqrt((sum_of_squared_residuals / (sample_size - 2)) / np.sum((X.T[0] - np.mean(X.T[0])) ** 2))
        standard_error_intercept = standard_error_slope * np.sqrt(np.sum(X.T[0] ** 2) / sample_size)
        squared_errors = (y - predictions) ** 2
        mean_squared_error = np.mean(squared_errors)
        standard_deviation = np.sqrt(mean_squared_error)
        confidence_interval_95 = 1.96 * (standard_deviation / np.sqrt(sample_size))

        # print("Predictions: ", predictions)
        # print("Samples: ", y)
        # print("Residual sum of squares: ", residual_sum_of_squares)
        # print("Total sum of squares: ", total_sum_of_squares)
        # print("Squared errors: ", squared_errors)
        # print(f"Mean squared error: {mean_squared_error}")
        print(f"Standard deviation: {standard_deviation}")
        print(f"Correlation coefficient: {correlation_coefficient}")
        print(f"R²: {r_squared}")
        # print(f"Variance: {}", np.sum((y - predictions) ** 2) / (sample_size - 2))
        print(f"Model standard error: {regression_standard_error}")
        print(f"Slope standard error: {standard_error_slope}")
        print(f"Intercept standard error intercept: {standard_error_intercept}")
        print(f"Confidence interval 95%: ±{confidence_interval_95}")

        plt.scatter(data[:, 0], data[:, 1])
        xs_predictions = np.reshape(np.linspace((min(data[:, 0])), max(data[:, 0]), 100), (100, 1))
        ys_predictions = model.predict(xs_predictions)
        plt.plot(xs_predictions, ys_predictions)
        plt.xlabel('Mileage')
        plt.ylabel('Price')
        plt.title('Price per mileage')
        plt.show()


if __name__ == '__main__':
    main()