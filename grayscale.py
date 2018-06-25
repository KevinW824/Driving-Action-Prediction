import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import os
import os.path
from pathlib import Path

# Create folder to save frames
try:
    if not os.path.exists('data'):
        os.makedirs('data')
except OSError:
    print ('Error: Creating directory of data')

# Initialize webcam
cap = cv2.VideoCapture('project_video.mp4')

currentFrame = 0

while cap.isOpened():
    my_file = Path('./data/frame' + str(currentFrame) + '.jpg')

    # time.sleep(0.3)
    # Read webcam image
    ret, frame = cap.read()

    # Convert image from RBG/BGR to GRAYSCALE
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Binary Thresholding
    # Set Threshold to 190-255
    # Everything out of the range will be displayed black
    ret,thresh = cv2.threshold(gray_img,195,255,cv2.THRESH_BINARY)


    # Display result
    cv2.imshow('Original', frame)
    cv2.imshow('grayimage', gray_img)
    cv2.imshow('thresh', thresh)

    # Save each frames
    # name = './data/frame' + str(currentFrame) + '.jpg'
    # print('Creating...' + name)
    # cv2.imwrite(name, thresh)

    # currentFrame += 1



    if cv2.waitKey(1) == 13:
        break



cap.release()
cv2.destroyAllWindows()