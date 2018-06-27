import glob
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import os
import os.path
from pathlib import Path
np.set_printoptions(threshold=np.inf)

im_thresh = cv2.imread('./data/frame0_thresh.jpg')
im_origin = cv2.imread('./data/frame0_frame.jpg')
im_gray = cv2.imread('./data/frame0_gray.jpg')
img_thresh = im_thresh[430:665,:]
img_origin = im_origin[430:665,:]
img_gray = im_gray[430:665,:]

def gray_scale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def binary_threshold(img):
    ret,img = cv2.threshold(img,175,255,cv2.THRESH_BINARY)
    return img


def gaussian_blur(img):
    img = cv2.GaussianBlur(img, (7,7), 0)
    return img

gimg = gray_scale(img_thresh)
timg = binary_threshold(gimg)

lst_row = timg[234:235,:]

index = []
for i in range(len(lst_row[0])):
    if lst_row[0][i] == 255:
        index.append(i)
print(index)

midpoint_left = (min(index) + max(index)) * 0.5
print(midpoint_left)


# col = np.sum(thresh, axis=0)
# convert = col/255
#
# plt.plot(convert)
# plt.show()

# ret,thresh = cv2.threshold(Gimg,175,255,cv2.THRESH_BINARY)
cv2.imshow('frame0_thresh', timg)
cv2.imshow('lst_raw',lst_row)

cv2.waitKey(0)
cv2.destroyAllWindows()
