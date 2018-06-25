import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import os
import os.path
from pathlib import Path


# def calc_line_fits_from_prevcalc_line(img, leftLine, rightLine):
#
#     left_fit = leftLine.best_fit_px
#     right_fit = rightLine.best_fit_px
#
#     ### Settings
#     margin = 100  # Width on either side of the fitted line to search
#
#     nonzero = img.nonzero()
#     nonzeroy = np.array(nonzero[0])
#     nonzerox = np.array(nonzero[1])
#
#     left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
#     nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
#     right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (
#     nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))
#
#     # Extract left and right line pixel positions
#     leftx = nonzerox[left_lane_inds]
#     lefty = nonzeroy[left_lane_inds]
#     rightx = nonzerox[right_lane_inds]
#     righty = nonzeroy[right_lane_inds]
#
#     # Fit a second order polynomial to each
#     left_fit = np.polyfit(lefty, leftx, 2)
#     right_fit = np.polyfit(righty, rightx, 2)
#
#     # Fit a second order polynomial to each in meters
#     left_fit_m = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
#     right_fit_m = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
#
#     # Generate x and y values for plotting
#     ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
#     left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
#     right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
#
#     # Create an image to draw on and an image to show the selection window
#     out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
#     window_img = np.zeros_like(out_img)
#
#     # Color in left and right line pixels
#     out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
#     out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
#
#     # Generate a polygon to illustrate the search window area
#     # And recast the x and y points into usable format for cv2.fillPoly()
#     left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
#     left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
#     left_line_pts = np.hstack((left_line_window1, left_line_window2))
#     right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
#     right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
#     right_line_pts = np.hstack((right_line_window1, right_line_window2))
#
#     # Draw the lane onto the warped blank image
#     cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
#     cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
#     result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
#
#     return left_fit, right_fit, left_fit_m, right_fit_m, result

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