import cv2
import numpy as np
from sklearn import cluster

image = cv2.imread("dados3.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edged = cv2.Canny(image, 100, 600)
cnts, hier = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for c in cnts:
    print(cv2.boundingRect(c))
idx = 0
for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    if w>60:
        idx+=1
        new_img = image[y:y+h, x:x+w]
        cv2.rectangle(image, (x, y), (x+w, y+h), (255,0,0), 1)

cv2.imshow('newimg',image)
cv2.waitKey(0)
cv2.destroyWindow('newimg')
cv2.waitKey(1)