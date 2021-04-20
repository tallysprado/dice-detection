from DiceCounter import Segmentation
import cv2

image1 = cv2.imread('dados1.jpg')
image2 = cv2.imread('dados2.jpg')
image3 = cv2.imread('dados3.png')

Segmentation.get_dices(image1, True)
Segmentation.get_dices(image2)
Segmentation.get_dices(image3)



