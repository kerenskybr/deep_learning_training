from nn.neuralnetwork import NeuralNetwork
import numpy as np

#Construct the XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

#Define our 2-2-1 neural network and train it
#2-2-1 means:
#1 - a input layer with 2 nodes
#2 - a single hidden layer with two nodes
#3 - An output layer with 1 node
nn = NeuralNetwork([2, 2, 1], alpha=0.5)
nn.fit(X, y, epochs=20000)

#Now that the network was trained
#loop over the XOR data points
for (x, target) in zip(X, y):
	#Make a prediction on the data point and
	#display the resul to our console
	pred = nn.predict(x)[0][0]
	step = 1 if pred > 0.5 else 0
	print("[INFO] data={}, ground-truth={}, pred={:.4f}, step={}".format(
		x, target[0], pred, step))