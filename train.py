import csv
import numpy as np
from linear_model import LinearModel


def main():
    with open('data.csv') as data_file:
        data_reader = csv.reader(data_file)
        raw_data = list(data_reader)
        # Slice to remove the first row (column names), convert to float and store in a numpy array
        data = np.array([[float(row[0]), float(row[1])] for row in raw_data[1:]])
        y = data[:, 1]
        X = data[:, 0:1]
        model = LinearModel()
        parameters, mean_error_history = model.fit(X, y, learning_rate=0.01, iterations=1000)
        with open('parameters', 'w') as out_file:
            out_file.write(', '.join(str(parameter) for parameter in parameters))


if __name__ == '__main__':
    main()