from DiceCounter import Segmentation
from DiceCounter import Utils
from DiceCounter import BlobDetector
import cv2
import numpy as np

image2 = cv2.imread('dados2.jpg')
image3 = cv2.imread('dados3.png')

image2 = cv2.blur(image2, (3,3))
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
image2 = Segmentation.threshold(image2, 220)
kernel = np.ones((4,4), np.uint8)
image2 = cv2.erode(image2, kernel, iterations=1)
keypoints, image = BlobDetector.count_by_blobs(image2)
image_no_filter = cv2.imread('dados2.jpg')
num, marked_image = BlobDetector.get_dice_from_blobs(keypoints, image_no_filter)
cv2.imshow('image', image)
cv2.imshow('image2', image2)
cv2.imshow('marked_image', marked_image)
'''
contour = Segmentation.image_with_contours(image2)
cv2.imshow('image2',contour)
#Utils.show(contour, 'contour')

contour = Segmentation.image_with_contours(image3)
cv2.imshow('image3',contour)
#Utils.show(contour, 'contour')
'''
cv2.waitKey(0)
cv2.destroyAllWindows()
