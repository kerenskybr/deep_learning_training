from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import argparse

def sigmoid_activation(x):
	#Compute the sigmoid activation vakue for a given input
	return 1.0 / (1 + np.exp(-x))

def predict(X, W):
	#take the dot product between our features and weight matrix
	preds = sigmoid_activation(X.dot(W))

	#apply a step function to threshold the outputs to binary
	#class labels
	preds[preds <= 0.5] = 0
	preds[preds > 0] = 1

	#return the predictions
	return preds

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=float, default=100,
	help="# of epochs")
ap.add_argument("-a", "--alpha", type=float, default=0.01,
	help="learning rate")
args=vars(ap.parse_args())

#Generate a 2-class classification problem with 1,000 data points
#where each data point is a 2D feature vector
(X, y) = make_blobs(n_samples=1000, n_features=2, centers=2,
	cluster_std=1.5, random_state=1)
y = y.reshape((y.shape[0], 1))

#insert a column of 1's as the last entry in the feature
#matrix -- this little trick allow us to treat the bias 
#as a trainable parameter within the weight matrix
#(the bias trick)
X = np.c_[X, np.ones((X.shape[0]))]

#Partition the data into training and testing splits using 50% of
#the data for training and the remaining 50% for testing
(trainX, testX, trainY, testY) = train_test_split(X, y,
	test_size=0.5, random_state=42)

#Iniatilize our weight matrix and list of losses
print("[INFO] training...")
#good initialization is critical to training a 
#neural network in a reasonable amount of time
W = np.random.randn(X.shape[1], 1)
losses = []

# loop over the desired number of epochs
for epoch in np.arange(0, args["epochs"]):
	#take hte dot product between our features "X" and the weight
	#matrix "W" then pass this values through our sigmoid activation
	#funtion thereby giving us our predicitons on the dataset
	preds = sigmoid_activation(trainX.dot(W))

	#Now tha we have our predictions, we need to determine
	#the "error", wich is the difference between our predicitons and
	#the true values
	error = preds - trainY
	loss = np.sum(error ** 2)
	losses.append(loss)

	#The gradient descent update is the dot product between 
	#our features and the error of the predictions
	gradient = trainX.T.dot(error)

	#In the update stage, all we need to do is "Nudge" the
	#weight matrix in the negative direction of the gradient (hence the 
	#term "gradient descent" ) by taking a small step towards a
	#set more optimal parameters
	W += -args["alpha"] * gradient

	#check to see if an update should be displayed
	if epoch == 0 or (epoch + 1) % 5 == 0:
		print("[INFO] epoch={}, loss={:.7f}".format(int(epoch + 1), loss))

#evaluate our model
print("[INFO] evaluating...")
preds = predict(testX, W)
print(classification_report(testY, preds))

'''
#plot the testing classification data
plt.style.use("ggplot")
plt.figure()
plt.title("Data")
plt.scatter(testX[:, 0], testX[:, 1], marker="o", c=testY, s=30)
'''	
#Construct a figure that plots the loss over time
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, args["epochs"]), losses)
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()