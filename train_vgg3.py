# USAGE
# python train_vgg.py --dataset animals --model output/smallvggnet.model --label-bin output/smallvggnet_lb.pickle --plot output/smallvggnet_plot.png

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
from pyimagesearch.smallvggnet import SmallVGGNet
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset of images")
'''
ap.add_argument("-m", "--model", required=True,
	help="path to output trained model")
ap.add_argument("-l", "--label-bin", required=True,
	help="path to output label binarizer")
ap.add_argument("-p", "--plot", required=True,
	help="path to output accuracy/loss plot")
'''    
args = vars(ap.parse_args())

# initialize the data and labels
print("[INFO] loading images...")
data = []
labels = []

# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images(args["dataset"])))
#random.seed(42)
#random.shuffle(imagePaths)

# loop over the input images
for imagePath in imagePaths:
	# load the image, resize it to 64x64 pixels (the required input
	# spatial dimensions of SmallVGGNet), and store the image in the
	# data list
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (100, 100))
	data.append(image)

	# extract the class label from the image path and update the
	# labels list
	label = imagePath.split(os.path.sep)[-2]
	labels.append(label)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.25, random_state=42)

# convert the labels from integers to vectors (for 2-class, binary
# classification you should use Keras' to_categorical function
# instead as the scikit-learn's LabelBinarizer will not return a
# vector)
le = LabelEncoder()
le.fit(labels)
print(le.classes_)
#exit()
np.save('imageclasses_vgg.npy',le.classes_)
#lb = LabelBinarizer()
trainY = le.transform(trainY)
testY = le.transform(testY)
labels=le.transform(labels)

# construct the image generator for data augmentation
#aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	#height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	#horizontal_flip=True, fill_mode="nearest")

# initialize our VGG-like Convolutional Neural Network
#model = SmallVGGNet.build(width=64, height=64, depth=3,
	#classes=len(lb.classes_))
model = Sequential()
inputShape = (100, 100, 3)
chanDim = -1
# CONV => RELU => POOL layer set
model.add(Conv2D(32, (3, 3), padding="same",
input_shape=inputShape))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))
 
model.add(Dense(len(le.classes_), activation="softmax"))
# softmax classifier
#model.add(Dense(len(le.classes_)))
#model.add(Dense(1))
#model.add(Activation("softmax"))
 
# initialize our initial learning rate, # of epochs to train for,
# and batch size
INIT_LR = 0.01
EPOCHS = 75
BS = 32

# initialize the model and optimizer (you'll want to use
# binary_crossentropy for 2-class classification)
print("[INFO] training network...")
opt = SGD(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="sparse_categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the network
#H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
#validation_data=(testX, testY), steps_per_epoch=len(trainX) ,epochs=EPOCHS)

#H = model.fit(data, labels, validation_data=(testX, testY),epochs=1, batch_size=32)
H = model.fit(data, labels,epochs=300, batch_size=32,verbose=1)
# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict_classes(testX, batch_size=32)
print(predictions)
#print(classification_report(testY.argmax(axis=1),
	#predictions.argmax(axis=1), target_names=le.classes_))

# plot the training loss and accuracy

# save the model and label binarizer to disk
model_json = model.to_json()
with open("image_test1_vgg.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("image_testw_vgg.h5")
print("Saved model to disk")
predictions = model.predict_classes(data)
# summarize the first 5 cases
for i in range(len(data)):
  print(str(predictions[i]) +" expected  "+str(labels[i])+str(le.inverse_transform([predictions[i]])))
# plot the training loss and accuracy
