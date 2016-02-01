from plot import *
from read_csv import read_csv

#  Basic configuration for LinearRegression-object.
BASIC_CONFIG = {
    'learning_rate': 0.5,
    'bias': 1,
    'num_of_epochs': 1500,
    'convergence_threshold': 1e-06
}


class LinearRegression(object):

    def __init__(self, config=BASIC_CONFIG):
        """
        Constructor for the LinearRegression-object
        :param config: A dictionary with object-configurations
        """
        self._config = BASIC_CONFIG
        self._config.update(config)
        self._weights = None
        self._bias = None
        print('# LinearRegression-object initialized')
        print('# Learning rate: %s, Maximum epochs: %s Convergence Threshold: %s' % (
            self._config['learning_rate'],
            self._config['num_of_epochs'],
            self._config['convergence_threshold'])
        )

    def gradient_descent(self, x_params, y_params, test_data=None):
        """
        Performes the gradient descent on the entire provided dataset
        :param x_params: All x-vectors
        :param y_params: Expected output for all corresponding x-vectors
        :return: adjusted weights, adjusted bias, and the history of the loss function
        """
        n, feature_count = len(y_params), x_params.shape[1]
        theta, bias = np.random.uniform(size=feature_count, low=.00001), self._config['bias']
        loss_history = []
        converged = False
        print('# Initial weights and bias: W%s B: %s\n' % (theta, bias))

        previous_prediction = self.cost(x_params, y_params, theta, bias)

        epochs = 0
        while epochs < self._config['num_of_epochs'] and not converged:
            epochs += 1

            for i in range(n):
                x_i, y_i = x_params[i], y_params[i]
                bias = bias + -self._config['learning_rate'] * ((2 / n) * (self.hypothesis(x_i, theta, bias) - y_i))
                theta = theta + -self._config['learning_rate'] * ((2 / n) * (x_i * (self.hypothesis(x_i, theta, bias) - y_i)))

            prediction = self.cost(x_params, y_params, theta, bias)

            if epochs == 5 or epochs == 10:
                temp_cost_training = self.cost(*training_data, theta, bias)
                temp_cost_test = None
                if test_data:
                    temp_cost_test = self.cost(*test_data, theta, bias)
                print('# Intermediate result of loss function after %i epochs: %s / %s (training/test)' % (epochs, temp_cost_training, temp_cost_test))

            loss_history.append(prediction)

            if abs(previous_prediction - prediction) <= self._config['convergence_threshold']:
                print('# Convergence reached after %i epochs' % epochs)
                converged = True

            previous_prediction = prediction

            if epochs == self._config['num_of_epochs']:
                print('# Maximum number of epochs has been reached')

        self._weights, self._bias = theta, bias
        return theta, bias, loss_history

    def cost(self, x_params, y_params, weights, bias):
        """
        Calculates MSE (Mean Squared Error) for the entire provided set
        :param x_params: Matrix containing all vectors of features
        :param y_params: Matrix containing vectors with expected output for all x
        :param weights: A vector of the current weights
        :param bias: The current bias as a integer
        :return: The MSE for the provided set as a numpy array with a single value
        """
        n = len(y_params)
        sum_ = sum([pow(self.hypothesis(x_params[i], weights, bias) - y_params[i], 2)
                    for i in range(n)])
        return sum_ * (1 / n)

    @staticmethod
    def hypothesis(x_i, weights, bias):
        """
        The linear regression model
        :param x_i: The features to calculate an output value for
        :param weights: The weights to use
        :param bias: The bias to use
        :return: A numpy array with the output value
        """
        return bias + np.dot(x_i, weights)

    def predict(self, x_i):
        """
        Predict the output value for a given input vector x_i
        :param x_i: The input vector
        :return: The predicted output value
        """
        if self._weights is None:
            raise Exception('# Unable to predict; weights and bias not set')
        return self.hypothesis(x_i, self._weights, self._bias)

    def predict_multiple(self, x_params):
        """
        Predict the output value for a set of size n
        :param x_params: The matrix containing x-vectors
        :return: A numpy array of all predicted values
        """
        predicted = np.array()
        for i in x_params:
            np.append(predicted, self.predict(x_params[i]))
        return predicted


if __name__ == "__main__":
    training_data = read_csv(training=True)
    test_data = read_csv(training=False)

    lr = LinearRegression()
    theta, bias, loss_history = lr.gradient_descent(*training_data, test_data=test_data)
    print('# Weights and bias after training: W%s, B%s' % (theta, bias))

    cost_training = lr.cost(*training_data, theta, bias)
    cost_test = lr.cost(*test_data, theta, bias)

    print('# Error: %s / %s (training/test)' % (cost_training, cost_test))

    plot_loss_history(loss_history, save=True)
    plot_regression(*training_data, lr, save=True)
