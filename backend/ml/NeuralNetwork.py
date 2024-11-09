import numpy as np
import pandas as pd
from scipy.special import expit, logit


class NeuralNetwork:
    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
        self.inodes = inputNodes
        self.hnodes = hiddenNodes
        self.onodes = outputNodes
        self.wih = np.random.normal(loc=0.0, scale=np.power(self.hnodes, - 1 / 2), size=(self.hnodes, self.inodes))
        self.who = np.random.normal(loc=0.0, scale=np.power(self.onodes, - 1 / 2), size=(self.onodes, self.hnodes))
        self.lr = learningRate
        self.activationFunction = expit
        self.inverseActivationFunction = logit

    def fit(self, X, y):
        outputNodes = len(y.unique())
        for i in range(0, len(X)):
            all_values = X.iloc[i, :]
            # масштабировать и сместить входные значения
            inputs = np.asfarray([float(i) for i in all_values])
            # создать целевые выходные значения (все равны 0,01, за исключением
            # желаемого маркерного значения, равного 0,99)
            targets = np.zeros(outputNodes) + 0.01
            # all_values[0] - целевое маркерное значение для данной записи
            targets[int(y[i])] = 0.99
            self.train(inputs, targets)

    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)

        hidden_outputs = self.activationFunction(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activationFunction(final_inputs)

        output_errors = targets - final_outputs

        hidden_errors = np.dot(self.who.T, output_errors)

        self.who += self.lr * np.dot(
            (output_errors * final_outputs * (1.0 - final_outputs)),
            np.transpose(hidden_outputs)
        )

        self.wih += self.lr * np.dot(
            (hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
            np.transpose(inputs)
        )

    def predict(self, X):
        predictions = []
        for i in range(0, len(X)):
            all_values = X.iloc[i, :]
            # print(correct_label, "истинный маркер")
            # масштабировать и сместить входные значения
            inputs = np.asfarray([float(i) for i in all_values])
            # опрос сети
            outputs = self.query(inputs)
            # индекс наибольшего значения является маркерным значением
            label = np.argmax(outputs)
            predictions.append(label)
        return pd.Series(predictions, index=X.index, name='labels')

    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)

        hidden_outputs = self.activationFunction(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activationFunction(final_inputs)

        return final_outputs

    # backquery the neural network
    # we'll use the same termnimology to each item,
    # eg target are the values at the right of the network, albeit used as input
    # eg hidden_output is the signal to the right of the middle nodes
    def backquery(self, targets_list):
        # transpose the targets list to a vertical array
        final_outputs = np.array(targets_list, ndmin=2).T

        # calculate the signal into the final output layer
        final_inputs = self.inverseActivationFunction(final_outputs)

        # calculate the signal out of the hidden layer
        hidden_outputs = np.dot(self.who.T, final_inputs)
        # scale them back to 0.01 to .99
        hidden_outputs -= np.min(hidden_outputs)
        hidden_outputs /= np.max(hidden_outputs)
        hidden_outputs *= 0.98
        hidden_outputs += 0.01

        # calculate the signal into the hidden layer
        hidden_inputs = self.inverseActivationFunction(hidden_outputs)

        # calculate the signal out of the input layer
        inputs = np.dot(self.wih.T, hidden_inputs)
        # scale them back to 0.01 to .99
        inputs -= np.min(inputs)
        inputs /= np.max(inputs)
        inputs *= 0.98
        inputs += 0.01

        return inputs
