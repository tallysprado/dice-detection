import cv2
import numpy as np

def count_dots(image):
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 255
    params.filterByArea = True
    params.minArea = 3
    params.maxArea = 400

    #disable the default settings
    params.filterByInertia = False
    params.filterByConvexity = False
    params.filterByCircularity = True
    params.minCircularity = 0. # 0.7 could be rectangular, too. 1 is round. Not set because the dots are not always round when they are damaged, for example.
    params.maxCircularity = 3.4028234663852886e+38 # infinity.
    params.filterByConvexity = False
    params.minConvexity = 0.
    params.maxConvexity = 3.4028234663852886e+38

    params.filterByInertia = True # a second way to find round blobs.
    params.minInertiaRatio = 0.55 # 1 is round, 0 is anywhat 
    params.maxInertiaRatio = 3.4028234663852886e+38 # infinity again

    params.minThreshold = 50 # from where to start filtering the image
    params.maxThreshold = 255.0 # where to end filtering the image
    params.thresholdStep = 5 # steps to go through
    params.minDistBetweenBlobs = 2 # avoid overlapping blobs. must be bigger than 0. Highly depending on image resolution! 
    params.minRepeatability = 2
    
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(image)
    img_with_keypoints = cv2.drawKeypoints(
        image, keypoints, np.array([]), (255,0,0),
        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    return keypoints, img_with_keypoints