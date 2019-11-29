from nn.perceptron import Perceptron
import numpy as np

#Construct the XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

#Define our perceptron and train it
print("[INFO] training perceptron...")
p = Perceptron(X.shape[1], alpha=0.1)
p.fit(X, y, epochs=20)

#Now evaluate the perceptron
print("[INFO] testing perceptron...")

#Now that our network is trained, loop over the data points
for (x, target) in zip(X, y):
	#Make a prediction on the data point and
	#display the result to our console
	pred = p.predict(x)
	print("[INFO] data={}, ground-truth={}, pred={}".format(
		x, target[0], pred))