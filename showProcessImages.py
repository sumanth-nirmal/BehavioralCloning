#!/usr/bin/python
## Author: sumanth
## Date: Feb, 05,2017
# test file to show the processed images
import matplotlib.pyplot as plt
import numpy as np
import processData
import pandas as pd

data = pd.read_csv(processData.dataPath)
img = plt.imread(processData.imPath+data.iloc[10][processData.center].strip())
angle = data.iloc[10][processData.steering]

# shear image
image, steering_angle = processData.randomShear(img, angle)
plt.imshow(image)
plt.show()

# crop image
image, steering_angle = processData.randomShear(img, angle)
image = processData.crop(img, 0.2, 0.1)
plt.imshow(image)
# flip the image
image, steering_angle = processData.randomFlip(img, angle)
plt.imshow(image)
# random gamma
image = processData.randomGamma(img)
plt.imshow(image)
# resize
image = processData.resize(img, 30)
plt.imshow(image)
