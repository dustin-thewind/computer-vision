from neural_net.nn import NeuralNetwork
import numpy as np

# construct XOR dataset
X = np.array([[0.71, 0.91], [0.73, 0.37], [0.32, 0.23]])
y = np.array([[0.94], [0.22], [0.49]])

# define a 2-2-1 NN and train it
nn = NeuralNetwork([2, 2, 1], alpha=0.5)
nn.fit(X, y, epochs=20000)

# loop over the XOR datasets
for (x, target) in zip(X, y):
    # make a prediction on the data point and display the result
    # to the console
    pred = nn.predict(x)[0][0]
    step = 1 if pred > 0.5 else 0
    print("[INFO] data={}, ground_truth={}, pred={:4f}, step={}".format(
        x, target[0], pred, step))
