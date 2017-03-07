#!/usr/bin/python
## Author: sumanth
## Date: Feb, 05,2017
# process the data into keras

"""
This script will process the data images.
images are saved as pickle and split for training and validation
"""
import argparse
import os
import sys
import csv
import base64
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cross_validation import train_test_split
import pickle

### Paths to folder and label
fPath = "/home/sumanth/catkin_ws/src/BehavioralCloning/trainingData/Udata/data"
csvPath = "{}/driving_log.csv".format(fPath)


### Import data
data = [] # empty list
with open(csvPath) as F:
    reader = csv.reader(F)
    for i in reader:
        data.append(i)

print("data imported")
### size of the data
print("data size:", len(data))

### Emtpy generators for feature and labels
features = ()
labels = ()

### This function will resize the images from front, left and
### right camera to 18 x 80 and turn them into lists.
### The length of the each list will be 18 x 80 = 1440
### j = 0,1,2 corresponds to center, left, right
def load_image(data_line, j):
    print(data_line[j].strip())
    img = plt.imread(fPath+'/'+data_line[j].strip())[65:135:4,0:-1:4,0]
    lis = img.flatten().tolist()
    return lis

#data = data[:100]
# For each item in data, convert camera images to single list
# and save them into features list.
for i in tqdm(range(int(len(data))), unit='items'):
    for j in range(3):
        features += (load_image(data[i],j),)


item_num = len(features)
print("features size", item_num)

# A single list will be convert back to the original image shapes.
# Each list contains 3 images so the shape of the result will be
# 54 x 80 where 3 images aligned vertically.
features = np.array(features).reshape(item_num, 18, 80, 1)
print("features shape", features.shape)

### Save labels
for i in tqdm(range(int(len(data))), unit='items'):
    for j in range(3):
        labels += (float(data[i][3]),)

labels = np.array(labels)

print("features:", features.shape)
print("labels:", labels.shape)

# Get randomized datasets for training and test
X_train, X_test, y_train, y_test = train_test_split(
    features,
    labels,
    test_size=0.10,
    random_state=123)

# Get randomized datasets for training and validation
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train,
    y_train,
    test_size=0.25,
    random_state=123)

# Print out shapes of new arrays
train_size = X_train.shape[0]
test_size = X_test.shape[0]
valid_size = X_valid.shape[0]
input_shape = X_train.shape[1:]
features_count = X_train.shape[1]*X_train.shape[2]*X_train.shape[3]

print("train size:", train_size)
print("valid size:", valid_size)
print("test size:", test_size)
print("input_shape:", input_shape)
print("features count:", features_count)

# Save the data for easy access
pickle_file = 'camera.pickle'
stop = False

while not stop:
    if not os.path.isfile(pickle_file):
        print('Saving data to pickle file...')
        try:
            with open(pickle_file, 'wb') as pfile:
                pickle.dump(
                    {
                        'train_dataset': X_train,
                        'train_labels': y_train,
                        'valid_dataset': X_valid,
                        'valid_labels': y_valid,
                        'test_dataset': X_test,
                        'test_labels': y_test,
                    },
                    pfile, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print('Unable to save data to', pickle_file, ':', e)
            raise

        print('Data cached in pickle file.')
        stop = True
    else:
        print("Please use a different file name other than camera.pickle")
        pickle_file = input("Enter: ")
