import cv2
import numpy as np
#import time

# Initialize webcam
cap = cv2.VideoCapture('project_video.mp4')

while cap.isOpened():

    # time.sleep(0.3)
    # Read webcam image
    ret, frame = cap.read()

    # Convert image from RBG/BGR to GRAYSCALE
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Set Threshold to 190-255, everything out of the range will be displayed black
    ret,thresh = cv2.threshold(gray_img,190,255,cv2.THRESH_BINARY)

    # Display result
    cv2.imshow('grayimage', gray_img)
    cv2.imshow('thresh', thresh)
    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()