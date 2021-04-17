from DiceCounter import Segmentation
from DiceCounter import Utils
from DiceCounter import BlobDetector
import cv2

image2 = cv2.imread('dados2.jpg')
image3 = cv2.imread('dados3.png')

contour = Segmentation.image_with_contours(image2)
Utils.show(contour, 'contour')

contour = Segmentation.image_with_contours(image3)
Utils.show(contour, 'contour')

