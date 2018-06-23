import cv2
import numpy as np
import time

# Initialize webcam
cap = cv2.VideoCapture('project_video.mp4')

# # define range of White color in HSL
#
# lower_white = np.array([0,0,0], dtype=np.uint8)
# upper_white = np.array([0,100,0], dtype=np.uint8)

while cap.isOpened():

    # time.sleep(0.3)
    # Read webcam image
    ret, frame = cap.read()

    # Convert image from RBG/BGR to GRAYSCALE
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray_img,190,255,cv2.THRESH_BINARY)

    # Use inRange to capture only the values between lower & upper_blue
    # mask = cv2.inRange(hsl_img, lower_white, upper_white)

    # Perform Bitwise AND on mask and our original frame
    # res = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('grayimage', gray_img)
    cv2.imshow('thresh', thresh)
    # cv2.imshow('mask', mask)
    # cv2.imshow('Filtered Color Only', res)
    if cv2.waitKey(1) == 13:
        break

cap.release()

cv2.destroyAllWindows()