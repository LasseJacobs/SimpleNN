
import numpy
import scipy.special

class NeuralNetwork:

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # set nodes
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes

        # set learning rate
        self.lr = learning_rate

        self.w_input_hidden = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.w_hidden_output = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # the activation function
        self.sigmoid = lambda x: scipy.special.expit(x)

        pass

    def train(self, inputs_list, targets_list):
        # convert input list to 2D array
        inputs = numpy.array(inputs_list, ndmin=2).T

        # calc signals into the hidden layer
        hidden_inputs = numpy.dot(self.w_input_hidden, inputs)
        # calc the signals coming from the hidden layer
        hidden_outputs = self.sigmoid(hidden_inputs)

        # calc signals into the output layer
        final_inputs = numpy.dot(self.w_hidden_output, hidden_outputs)
        final_outputs = self.sigmoid(final_inputs)

        targets = numpy.array(targets_list, ndim=2).T
        # calc the errors
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.w_hidden_output.T, output_errors)

        self.w_hidden_output += self.lr * numpy.dot((output_errors * final_outputs * (1 - final_outputs)),
                                                    numpy.transpose(hidden_outputs))
        self.w_hidden_output += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)),
                                                    numpy.transpose(inputs_list))

        pass

    def querry(self, inputs_list):
        # convert input list to 2D array
        inputs = numpy.array(inputs_list, ndmin=2).T

        # calc signals into the hidden layer
        hidden_inputs = numpy.dot(self.w_input_hidden, inputs)
        # calc the signals coming from the hidden layer
        hidden_outputs = self.sigmoid(hidden_inputs)

        # calc signals into the output layer
        final_inputs = numpy.dot(self.w_hidden_output, hidden_outputs)

        final_outputs = self.sigmoid(final_inputs)

        return final_outputs




NN = NeuralNetwork(3, 3, 3, 0.3)

result = NN.querry([1.0, 0.5, -1.5])
print result