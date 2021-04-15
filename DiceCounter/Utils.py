import cv2

def show(image, name):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyWindow(name)
    cv2.destroyAllWindows()