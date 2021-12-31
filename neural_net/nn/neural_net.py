#import packages
import numpy as np

class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):
        # init the list of weights matrices, then store the
        # network architecture and learning rate
        self.W = []
        self.layers = layers
        self.alpha = alpha

        # start loopiong from the index of the first layer
        # but stop before we reach the last two layers
        for i in np.arange(0, len(layers) - 2):
            # randomly init a weight matrix connecting the
            # number of nodes in each respective layer together
            # adding an extra node for the bias
            w = np.random.randn(layers[i] + 1, layers[i + 1] + 1)
            self.W.append(w / np.sqrt(layers[i]))

        # the last two layers are a special case where the input
        # connections need a bias term but the output does not
        w = np.random.randn(layers[-2] + 1, layers[-1])
        self.W.append(w / np.sqrt(layers[-2]))

    def __repr__(self):
        # construct and return a string that represents the
        # network architecture
        return "NeuralNetwork: {}".format(
            "-".join(str(l) for l in self.layers))

    def sigmoid(self, x):
        # compute and return the sigmoid activation value
        # for a given input value
        return 1.0 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        # compute the derivative of the sigmoid function
        # assuming that x has already been passed through the
        # sigmoid function
        return x * (1 - x)

    def fit(self, X, y, epochs=1000, displayUpdate=100):
        # insert a column of 1's as the last entry in the feature
        # matrix - this allows us to treat the bias as a
        # trainable parameter within the weight matrix
        X = np.c_[X, np.ones((X.shape[0]))]

        # loop over the desired number of epochs
        for epoch in np.arange(0, epochs):
            # loop over each individual data point and train
            # our network on it
            for (x, target) in zip(X, y):
                self.fit_partial(x, target)

            # check to see if we should display a training update
            if epoch == 0 or (epoch + 1) % displayUpdate == 0:
                loss = self.calculate_loss(X, y)
                print("[INFO] epoch={}, loss={:.7f}".format(
                    epoch + 1, loss))

    def fit_partial(self, x, y):
        # construct our list of output activations for each layer
        # as our data point flows through the network; the first
        # activation is a special case as its just the input
        # feature vector itself
        A = [np.atleast_2d(x)]

        # FEEDFORWARD:

        # loop over the layers in the network
        for layer in np.arange(0, len(self.W)):
            # feedforward the activation at the current layer by
            # taking the dot product between the activation and
            # the weight matrix -- this is the "net input"
            # the current layer
            net = A[layer].dot(self.W[layer])

            # computing the "net output" is simply applying
            # our non-liner activation function to the net input
            out = self.sigmoid(net)

            # once we have the net output, add it to our list of
            # activations
            A.append(out)

        # BACKPROPAGATION
        # the first phase of backpropagation is to compute the
        # difference between our prediction and the true
        # target value
        error = A[-1] - y

        # now we apply the chain rule and build our
        # list of deltas 'D'; the first entry in the deltas
        # is simply the error of the output layer times the
        # derivative of our activation function for the output value
        D = [error * self.sigmoid_deriv(A[-1])]

        # implement the chain with by looping over the layers
        # in reverse order (ignoring the last 2 layers since
        # we already handled them)
        for layer in np.arange(len(A) - 2, 0, -1):
            # the delta for the current layer is equal to the
            # delta of the previous layer dotted with the weight matrix
            # of the current layer, followed by multiplying the delta
            # by the derivative of the non-linear activation fuction
            # for the activations of the current layer
            delta = D[-1].dot(self.W[layer].T)
            delta = delta * self.sigmoid_deriv(A[layer])
            D.append(delta)

        # since we looped over our layers in reverse order
        # we need to reverse the deltas
        D = D[::-1]

        # WEIGHT UPDATE
        # loop over the layers
        for layer in np.arange(0, len(self.W)):
            # update the weights by taking the dot product of the layer
            # activations with their respective deltas, then multiplying
            # this value by some small learning rate and adding to our
            # weight matrix - this is where the actual learning takes place
            self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])

    def predict(self, X, addBias=True):
        # init the output prediction as the input features
        # then forward propagate this value through the network
        # to obtain a final prediction
        p = np.atleast_2d(X)

        # check to see if the bias column should be added
        if addBias:
            # insert a columne of 1's as the last entry
            # in the feature matrix (bias)
            p = np.c_[p, np.ones((p.shape[0]))]


        # loop over our layers in the network
        for layer in np.arange(0, len(self.W)):
            # we compute the output prediction
            # by taking the dot product between the current activation value
            # 'p' and the weight matrix associated with the current layer
            # then passing this value through a nonliner activation function
            p = self.sigmoid(np.dot(p, self.W[layer]))

        # return the predicted value
        return p

    def calculate_loss(self, X, targets):
        # make predictions for the input data points
        # then compute the loss
        targets = np.atleast_2d(targets)
        predictions = self.predict(X, addBias=False)
        loss = 0.5 * np.sum((predictions - targets) ** 2)

        # return the loss
        return loss
