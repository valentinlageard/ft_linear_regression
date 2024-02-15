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
        model = LinearModel()
        iterations = 1000
        parameters, mean_error_history = model.fit(X, y, learning_rate=0.01, iterations=iterations)
        with open('parameters', 'w') as out_file:
            out_file.write(', '.join(str(parameter) for parameter in parameters))
        
        fig, axs = plt.subplots(2)
        xs = np.arange(iterations)
        axs[0].plot(xs, mean_error_history)
        axs[0].set(xlabel='Iteration', ylabel='Mean error (standardized)')
        axs[0].set_title('Learning curve')
        axs[1].scatter(data[:, 0], data[:, 1])
        xs_predictions = np.reshape(np.linspace((min(data[:, 0])), max(data[:, 0]), 100), (100, 1))
        ys_predictions = model.predict(xs_predictions)
        axs[1].plot(xs_predictions, ys_predictions)
        axs[1].set(xlabel='Mileage', ylabel='Price')
        axs[1].set_title('Price per mileage')
        plt.show()


if __name__ == '__main__':
    main()