import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

def sigmoid_activation(x):
    #Compute the sigmoid activation value for a given input
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

#X is the feature of the dataset (image pixels)
#y is the label class
#batchSize - the size of each batch
def next_batch(X, y, batchSize):
    #loop over our dataset "X" in mini-batches, yielding a tuple
    #of the current batched data and labels
    for i in np.arange(0, X.shape[0], batchSize):
        yield (X[i:i + batchSize], y[i:i + batchSize])

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=float, default=100, help="# of epochs")
ap.add_argument("-a", "--alpha", type=float, default=0.01, help="learning rate")
ap.add_argument("-b", "--batch-size", type=int, default=32, help="size of SGD mini-batches")
args = vars(ap.parse_args())

#generate a 2 class classification problem with 1,000 data points,
#where each data points is a 2D feature vector
(X, y) = make_blobs(n_samples=1000, n_features=2, centers=2,cluster_std=1.5, random_state=1)
y = y.reshape((y.shape[0], 1))
#insert a column of 1's as the last entry in the feature
#matrix - this little trick allows us to treat the bias
#as a trainable parameter within the weight matrix
X = np.c_[X, np.ones((X.shape[0]))]

#Partition the data into training and testing splits using 50%
#of the data for training and 50% for testing
(trainX, testX, trainY, testY) = train_test_split(X, y, test_size=0.5,
    random_state=42)

#Initialize our weight matrix and list of losses
print("[INFO] training...")
W = np.random.randn(X.shape[1], 1)
losses = []

#Loop over the desired number of epochs
for epoch in np.arange(0, args["epochs"]):
    #initialize the total loss for the epoch
    epochLoss = []

    #Loop over our data in batches
    for (batchX, batchY) in next_batch(X, y, args["batch_size"]):
        #take the dot product reween our current batch of features
        #and the weight matrix, then pass this value trought our
        #activation function
        preds = sigmoid_activation(batchX.dot(W))

        #now that we have our predicitions, we need to determine the
        #error, wich is the difference between our predictions
        #and the true values
        error = preds - batchY
        epochLoss.append(np.sum(error ** 2))

        #The gradient descent update is the dot product between
        #our currrent batch and the error on the batch
        gradient = batchX.T.dot(error)

        #In the update stage, all we need to do is nudge the
        #weight matrix in the negative direction of the gradient
        #*hence the term "gradient descent") by taking a small step
        #towards a set of "more optimal" parameters
        W += -args['alpha'] * gradient

        #update our loss history by taking the 
        #average loss across all batches
        loss = np.average(epochLoss)
        losses.append(loss)

        #Check to see if an update should be displayed
        if epoch == 0 or (epoch + 1) % 5 == 0:

            print("[INFO] epoch={}, loss{:.4f}".format(int(epoch + 1),loss))


