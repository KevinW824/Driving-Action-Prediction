import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

im = cv2.imread('./data/frame5.jpg')
ori = im[:665,:]

gray_img = cv2.cvtColor(ori, cv2.COLOR_BGR2GRAY)

ret,img = cv2.threshold(gray_img,195,255,cv2.THRESH_BINARY)
cv2.imshow('frame5', img)

def add_lines(img, src):
    img2 = np.copy(img)
    color = [255, 0, 0] # Red
    thickness = 2
    x0, y0 = src[0]
    x1, y1 = src[1]
    x2, y2 = src[2]
    x3, y3 = src[3]
    cv2.line(img2, (x0, y0), (x1, y1), color, thickness)
    cv2.line(img2, (x1, y1), (x2, y2), color, thickness)
    cv2.line(img2, (x2, y2), (x3, y3), color, thickness)
    cv2.line(img2, (x3, y3), (x0, y0), color, thickness)
    return img2


def add_point (img, src):
    img2 = np.copy(img)
    color = [255, 0, 0] # Red
    thickness = -1
    radius = 15
    x0, y0 = src[0]
    x1, y1 = src[1]
    x2, y2 = src[2]
    x3, y3 = src[3]
    cv2.circle(img2, (x0, y0), radius, color, thickness)
    cv2.circle(img2, (x1, y1), radius, color, thickness)
    cv2.circle(img2, (x2, y2), radius, color, thickness)
    cv2.circle(img2, (x3, y3), radius, color, thickness)
    return img2

src = np.float32([
    [225, 660],
    [590, 450],
    [695, 450],
    [1075, 660]
])
# Points for the new image
dst = np.float32([
    [400, 720],
    [400, 0],
    [880, 0],
    [880, 720]
])


def warper(img):
    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

    return warped


def unwarp(img):
    # Compute and apply inverse perpective transform
    img_size = (img.shape[1], img.shape[0])
    Minv = cv2.getPerspectiveTransform(dst, src)
    unwarped = cv2.warpPerspective(img, Minv, img_size, flags=cv2.INTER_NEAREST)

    return unwarped


warped = warper(img)

# add the points to the og and warped images
src_points_img = add_point(img, src)
src_points_img = add_lines(src_points_img, src)
dst_points_warped = add_point(warped, dst)
dst_points_warped = add_lines(dst_points_warped, dst)

### Plot the source points on the original image and the warped image
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(src_points_img)
ax1.set_title('Original Image with Source Points', fontsize=25)
ax2.imshow(dst_points_warped)
ax2.set_title('Warped Perspective', fontsize=25)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()

# points_A = np.float32([[575, 445], [655, 445], [340, 665], [1110, 665]])
#
# points_B = np.float32([[0, 0], [770, 0], [0, 220], [770, 220]])
#
# M = cv2.getPerspectiveTransform(points_A, points_B)
#
# warped = cv2.warpPerspective(thresh, M, (770, 220))
#
# cv2.imshow('warpPerspective', warped)

cv2.waitKey(0)
cv2.destroyAllWindows()
