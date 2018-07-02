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
img_thresh = im_thresh[430:665,:]

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

# cut the last row
lst_row = timg[234:235,:]

# Find where the white part is
index = []
for i in range(len(lst_row[0])):
    if lst_row[0][i] == 255:
        index.append(i)
print(index)

calculate the middle
midpoint_left = (min(index) + max(index)) * 0.5
print(midpoint_left)D


edges = cv2.Canny(timg,100,200)
lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
for rho,theta in lines[0]:
	a = np.cos(theta)
	b = np.sin(theta)
	x0 = a*rho
	y0 = b*rho
	x1 = int(x0 + 1000*(-b))
	y1 = int(y0 + 1000*(a))
	x2 = int(x0 - 1000*(-b))
	y2 = int(y0 - 1000*(a))
	cv2.line(img_thresh,(x1,y1),(x2,y2),(0,0,255),3)

# ret,thresh = cv2.threshold(Gimg,175,255,cv2.THRESH_BINARY)
cv2.imshow('frame0_thresh', edges)
cv2.imshow('original', img_thresh)
# cv2.imshow('lst_raw',lst_row)

cv2.waitKey(0)
cv2.destroyAllWindows()
