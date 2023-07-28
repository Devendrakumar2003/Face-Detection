import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import cv2
from sklearn.model_selection import train_test_split
import pickle
import os
import pandas as pd
import random
from keras.preprocessing.image import ImageDataGenerator
path = "D:/aryan/Vit_temp/5th_sem/CSE3501_ISAA/Project/myData_2" #folder with all the class folders
labelFile = 'D:/aryan/Vit_temp/5th_sem/CSE3501_ISAA/Project/labels.csv' # file with all names of classes
batch_size_val = 40 # how many to process together
steps_per_epoch_val = 7
epochs_val = 25
imageDimesions = (180,320, 3)
testRatio = 0.2 # if 1000 images split will 200 for testing
validationRatio = 0.2 # if 1000 images 20% of remaining 800 will be 160 for validation
count = 0
images = []
classNo = []
myList = os.listdir(path)
print("Total Classes Detected:", len(myList))
noOfClasses = len(myList)
print("Importing Classes.....")
for x in range(0, len(myList)):
 myPicList = os.listdir(path + "/" + str(count))
 for y in myPicList:
 curImg = cv2.imread(path + "/" + str(count) + "/" + y)
 images.append(curImg)
 classNo.append(count)
 print(count, end=" ")
 count += 1
print(" ")
images = np.array(images)
classNo = np.array(classNo)
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validationRatio)
print("Data Shapes")
print("Train", end="");
print(X_train.shape, y_train.shape)
print("Validation", end="");
print(X_validation.shape, y_validation.shape)
print("Test", end="");
print(X_test.shape, y_test.shape)
assert (X_train.shape[0] == y_train.shape[0]), "The number of images in not equal to the number of lables in training set"
assert (X_validation.shape[0] == y_validation.shape[0]), "The number of images in not equal to the number of lables in 
validation set"
assert (X_test.shape[0] == y_test.shape[0]), "The number of images in not equal to the number of lables in test set"
assert (X_train.shape[1:] == (imageDimesions)), " The dimesions of the Training images are wrong "
assert (X_validation.shape[1:] == (imageDimesions)), " The dimesionas of the Validation images are wrong "
assert (X_test.shape[1:] == (imageDimesions)), " The dimesionas of the Test images are wrong"
data = pd.read_csv(labelFile)
print("data shape ", data.shape, type(data))
num_of_samples = []
cols = 20
num_classes = noOfClasses
fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(5, 300))
fig.tight_layout()
for i in range(cols):
 for j, row in data.iterrows():
 x_selected = X_train[y_train == j]
 axs[j][i].imshow(x_selected[random.randint(0, len(x_selected) - 1), :, :], cmap=plt.get_cmap("gray"))
 axs[j][i].axis("off")
 if i == 2:
 axs[j][i].set_title(str(j) + "-" + row["Name"])
 num_of_samples.append(len(x_selected))
print(num_of_samples)
plt.figure(figsize=(12, 4))
plt.bar(range(0, num_classes), num_of_samples)
plt.title("Distribution of the training dataset")
plt.xlabel("Class number")
plt.ylabel("Number of images")
plt.show()
def grayscale(img):
 img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 return img
def equalize(img):
 img = cv2.equalizeHist(img)
 return img
def preprocessing(img):
 img = grayscale(img) # CONVERT TO GRAYSCALE
 img = equalize(img) # STANDARDIZE THE LIGHTING IN AN IMAGE
 img = img / 255 # TO NORMALIZE VALUES BETWEEN 0 AND 1 INSTEAD OF 0 TO 255
 return img
X_train = np.array(list(map(preprocessing, X_train))) # TO IRETATE AND PREPROCESS ALL IMAGES
X_validation = np.array(list(map(preprocessing, X_validation)))
X_test = np.array(list(map(preprocessing, X_test)))
cv2.imshow("GrayScale Images",
 X_train[random.randint(0, len(X_train) - 1)]) # TO CHECK IF THE TRAINING IS DONE PROPERLY
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
dataGen = ImageDataGenerator(width_shift_range=0.1,
 # 0.1 = 10% IF MORE THAN 1 E.G 10 THEN IT REFFERS TO NO. OF PIXELS EG 10 PIXELS
 height_shift_range=0.1,
 zoom_range=0.2, # 0.2 MEANS CAN GO FROM 0.8 TO 1.2
 shear_range=0.1, # MAGNITUDE OF SHEAR ANGLE
 rotation_range=10) # DEGREES
dataGen.fit(X_train)
# REQUESTING DATA GENRATOR TO GENERATE IMAGES BATCH SIZE = NO. OF IMAGES CREAED EACH 
TIME ITS CALLED
batches = dataGen.flow(X_train, y_train,batch_size=20)
X_batch, y_batch = next(batches)
# TO SHOW AGMENTED IMAGE SAMPLES
fig, axs = plt.subplots(1, 15, figsize=(20, 5))
fig.tight_layout()
for i in range(10):
 axs[i].imshow(X_batch[i].reshape(imageDimesions[0], imageDimesions[1]))
 axs[i].axis('off')
plt.show()
y_train = to_categorical(y_train, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)
def myModel():
 no_Of_Filters = 30
 size_of_Filter = (5, 5) # THIS IS THE KERNEL THAT MOVE AROUND THE IMAGE TO GET THE FEATURES.
 # THIS WOULD REMOVE 2 PIXELS FROM EACH BORDER WHEN USING IMAGE
 size_of_Filter2 = (3, 3)
 size_of_pool = (2, 2) # SCALE DOWN ALL FEATURE MAP TO GERNALIZE MORE, TO REDUCE OVERFITTING
 no_Of_Nodes = 500 # NO. OF NODES IN HIDDEN LAYERS
 model = Sequential()
 model.add((Conv2D(no_Of_Filters, size_of_Filter, input_shape=(imageDimesions[0], imageDimesions[1], 1),
 activation='relu'))) # ADDING MORE CONVOLUTION LAYERS = LESS FEATURES BUT CAN CAUSE 
ACCURACY TO INCREASE
 model.add((Conv2D(no_Of_Filters, size_of_Filter, activation='relu')))
 model.add(MaxPooling2D(pool_size=size_of_pool)) # DOES NOT EFFECT THE DEPTH/NO OF FILTERS
 model.add((Conv2D(no_Of_Filters // 2, size_of_Filter2, activation='relu')))
 model.add((Conv2D(no_Of_Filters // 2, size_of_Filter2, activation='relu')))
 model.add(MaxPooling2D(pool_size=size_of_pool))
 model.add(Dropout(0.5))
 model.add(Flatten())
 model.add(Dense(no_Of_Nodes, activation='relu'))
 model.add(Dropout(0.5)) # INPUTS NODES TO DROP WITH EACH UPDATE 1 ALL 0 NONE
 model.add(Dense(noOfClasses, activation='softmax')) # OUTPUT LAYER
 # COMPILE MODEL
 model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
 return model
############################### TRAIN
model = myModel()
print(model.summary())
history = model.fit(dataGen.flow(X_train, y_train, batch_size=batch_size_val),
 steps_per_epoch=steps_per_epoch_val, epochs=epochs_val,
 validation_data=(X_validation, y_validation), shuffle=1)
############################### PLOT
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Acurracy')
plt.xlabel('epoch')
plt.show()
score = model.evaluate(X_test, y_test, verbose=0)
print('Test Score:', score[0])
print('Test Accuracy:', score[1])
frameWidth = 640 # CAMERA RESOLUTION
frameHeight = 480
brightness = 180
threshold = 0.8 # PROBABLITY THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX
# SETUP THE VIDEO CAMERA
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)
def grayscale(img):
 img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 return img
def equalize(img):
 img = cv2.equalizeHist(img)
 return img
def preprocessing(img):
 img = grayscale(img)
 img = equalize(img)
 img = img / 255
 return img
def getCalssName(classNo):
 if classNo == 0:
 return 'Aryan'
 elif classNo == 1:
 return 'Devendra'
 elif classNo == 2:
 return 'Tarang'
 elif classNo == 3:
 return 'Abhishek'
 else:
 return 'Unidentified user'
while True:
 success, imgOrignal = cap.read()
 # PROCESS IMAGE
 img = np.asarray(imgOrignal)
 img = cv2.resize(img, (180,320))
 img = preprocessing(img)
 cv2.imshow("Processed Image", img)
 img = img.reshape(1, 180,320, 1)
 cv2.putText(imgOrignal, "CLASS: ", (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
 cv2.putText(imgOrignal, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
 # PREDICT IMAGE
 predictions = model.predict(img)
 classIndex = np.argmax(predictions,axis=1)
 probabilityValue = np.amax(predictions)
 if probabilityValue > threshold:
 cv2.putText(imgOrignal, str(classIndex) + " " + str(getCalssName(classIndex)), (120, 35), font, 0.75, (0, 0, 255), 2,
 cv2.LINE_AA)
 cv2.putText(imgOrignal, str(round(probabilityValue * 100, 2)) + "%", (180, 75), font, 0.75, (0, 0, 255), 2, 
cv2.LINE_AA)
 cv2.imshow("Result", imgOrignal)
 if cv2.waitKey(1) and 0xFF == ord('q'):
 cv2.destroyAllWindows()
