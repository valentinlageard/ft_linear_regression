import os
import numpy as np
from linear_model import LinearModel


def main():
    parameters = np.zeros(2, dtype='float64')
    if os.path.exists('parameters'):
        with open('parameters') as file:
            parameters = np.array(file.read().split(', '), dtype='float64')
    x = None
    while True:
        try:
            x = np.reshape(np.array(input("Enter a mileage: "), dtype='float64'), (1, 1))
            break
        except ValueError as e:
            print(e)
    model = LinearModel()
    model.parameters = parameters
    prediction = model.predict(x)
    print(prediction[0])


if __name__ == '__main__':
    main()