from DiceCounter import Segmentation
import cv2

image2 = cv2.imread('dados2.jpg')
image3 = cv2.imread('dados3.png')

Segmentation.get_dices(image2)
Segmentation.get_dices(image3)

#Utils.show(contour, 'contour')



