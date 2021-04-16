from DiceCounter import Segmentation
from DiceCounter import Utils
from DiceCounter import BlobDetector
import cv2

image2 = cv2.imread('dados2.jpg')
image3 = cv2.imread('dados3.png')

threshold_image = Segmentation.threshold(image2, 50)
Utils.show(threshold_image, 'thresh')
contour = Segmentation.image_with_contours(image2)
Utils.show(contour, 'contour')

threshold_image = Segmentation.threshold(image3, 50)
Utils.show(threshold_image, 'thresh')
contour = Segmentation.image_with_contours(image3)
Utils.show(contour, 'contour')

