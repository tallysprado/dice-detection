import cv2
import numpy as np
from DiceCounter import BlobDetector

def threshold(img, value):
    out = img.copy()
    ret, out = cv2.threshold(out, value, 255, cv2.THRESH_BINARY)

    return out


def image_with_contours(image):
    #contours only works with canny images
    canny = cv2.Canny(image, 100, 550)
    out = image.copy()
    cnts, hier = cv2.findContours(
        canny.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w > 40:
            new_img = out[y:y+h, x:x+w]

            #draw dots counts
            dots, img = BlobDetector.count_dots(new_img)
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeft = (x,y)
            fontScale = 1
            fontColor = (255,255,255)
            lineType = 2
            cv2.putText(
                out, str(len(dots)), bottomLeft, font, fontScale, fontColor, lineType
            )            
            
            cv2.rectangle(out, (x, y), (x+w, y+h), (255, 0, 0), 1)
    return out