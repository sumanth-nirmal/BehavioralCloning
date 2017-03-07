#!/usr/bin/python
## Author: sumanth
## Date: Feb, 05,2017
# test file to show the processed images
import matplotlib.pyplot as plt
import numpy as np
import processData

i = 0
for j in range(3):
    plt.subplot(121)
    img = plt.imread(processData.fPath+'/'+processData.data[i][j].strip())
    plt.imshow(img)
    if j == 0:
        plt.title("Center")
    elif j == 1:
        plt.title("Left")
    elif j == 2:
        plt.title("Right")
    plt.subplot(122)
    a = np.array(processData.load_image(processData.data[i], j)).reshape(1, 18, 80, 1)
    print(a.shape)
    plt.imshow(a[0,:,:,0])
    plt.title("Resized")
    plt.show()
del(a)
