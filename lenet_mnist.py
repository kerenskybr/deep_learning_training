from nn.conv.lenet import LeNet
from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
from sklearn.datasets import fetch_openml
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np

#Grab the MNIST dataset
print("[INFO] loading MNIST dataset....")
dataset = fetch_openml("mnist_784")
data = dataset.data

if K.image_data_format() == "channels_first":
	data = data.reshape(data.shape[0], 1, 28, 28)

else:
	data = data.reshape(data.shape[0], 28, 28, 1)

(trainX, testX, trainY, testY) = train_test_split(data / 255.0,
	dataset.target.astype("int"), test_size=0.25, random_state=42)

#Convert the labels from integers to vectors
le = LabelBinarizer()
trainY = le.fit_transform(trainY)
testY = le.transform(testY)

#Initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=0.01)
model = LeNet.build(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

#Train the network
print("[INFO] training network")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
	batch_size=128, epochs=20, verbose=1)

#Evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1),
	target_names=[str(x) for x in le.classes_]))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 20), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 20), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 20), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 20), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()