import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.datasets import mnist
from keras.utils import np_utils
from util import load_model
from keras import backend as K
K.set_image_dim_ordering('th')

# fix a random seed for reproducibility
np.random.seed(9)

# user inputs
nb_epoch = 25
num_classes = 10
batch_size = 128
train_size = 60000
test_size = 10000
v_length = 784

# split the mnist data into train and test
(trainData, trainLabels), (testData, testLabels) = mnist.load_data()
print("[INFO] train data shape: {}".format(trainData.shape))
print("[INFO] test data shape: {}".format(testData.shape))
print( "[INFO] train samples: {}".format(trainData.shape[0]))
print("[INFO] test samples: {}".format(testData.shape[0]))

# reshape the dataset
trainData = trainData.reshape(train_size, v_length)
testData = testData.reshape(test_size, v_length)
trainData = trainData.astype("float32")
testData = testData.astype("float32")
trainData /= 255
testData /= 255

print( "[INFO] train data shape: {}".format(trainData.shape))
print( "[INFO] test data shape: {}".format(testData.shape))
print( "[INFO] train samples: {}".format(trainData.shape[0]))
print( "[INFO] test samples: {}".format(testData.shape[0]))

# convert class vectors to binary class matrices --> one-hot encoding
mTrainLabels = np_utils.to_categorical(trainLabels, num_classes)
mTestLabels = np_utils.to_categorical(testLabels, num_classes)

# grab some test images from the test data
test_images = testData
size = len(test_images)
org_image = testData[10:14]
org_image = org_image.reshape(org_image.shape[0],28, 28)
# reshape the test images to standard 28x28 format
test_images = test_images.reshape(test_images.shape[0], 1, 28, 28)
print("[INFO] test images shape - {}".format(test_images.shape))

model = load_model('simple_conv')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
axes =[0]*size
# # loop over each of the test images
# for i, test_image in enumerate(test_images, start=1):
# 	# grab a copy of test image for viewing
# 	#org_image = test_image
	
# 	# reshape the test image to [1x784] format so that our model understands
# 	test_image =test_image.reshape(test_image.shape[0], 1, 28, 28).astype('float32')

	
# 	# make prediction on test image using our trained model
# 	prediction = model.predict_classes(test_image, verbose=0)
	
# 	# display the prediction and image
# 	print("[INFO] I think the digit is - {}".format(prediction[0]))
# 	axes[i-1]=plt.subplot(220+i)

# 	plt.imshow(org_image[i-1], cmap=plt.get_cmap('gray'))

# 	axes[i-1].text(0.5,-0.1, str(prediction[0]), size=12, ha="center", 
#          transform=axes[i-1].transAxes, color='blue')

# 	#axes[i-1].set_xlabel(prediction[0])

# plt.show()

predicted_classes = model.predict_classes(test_images)
# see which we predicted correctly and which not
correct_indices = np.nonzero(predicted_classes == testLabels)[0]
incorrect_indices = np.nonzero(predicted_classes != testLabels)[0]
print(len(correct_indices)," classified correctly")
print(len(incorrect_indices)," classified incorrectly")

# adapt figure size to accomodate 18 subplots
plt.rcParams['figure.figsize'] = (7,14)

figure_evaluation = plt.figure()

# plot 9 correct predictions
for i, correct in enumerate(correct_indices[:9]):
    plt.subplot(6,3,i+1)
    plt.imshow(test_images[correct].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], testLabels[correct]))
    plt.xticks([])
    plt.yticks([])

# plot 9 incorrect predictions
for i, incorrect in enumerate(incorrect_indices[:9]):
    plt.subplot(6,3,i+10)
    plt.imshow(test_images[incorrect].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], testLabels[incorrect]))
    plt.xticks([])
    plt.yticks([])

plt.show()

