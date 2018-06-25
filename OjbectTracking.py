import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

im = cv2.imread('./data/frame5.jpg')
img = im[:665,:]

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret,thresh = cv2.threshold(gray_img,195,255,cv2.THRESH_BINARY)
cv2.imshow('frame5', thresh)

col = np.sum(thresh, axis=0)
convert = col/255

plt.plot(convert)
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
