import cv2
import numpy as np
from sklearn import cluster
from DiceCounter import Segmentation
from DiceCounter import Utils

def get_dice_from_blobs(blobs):
    X = []
    for b in blobs:
        position = b.pt
        if position != None:
            X.append(position)
    X = np.asarray(X)
    if len(X)>0:
        clustering = cluster.DBSCAN(eps=40, min_samples=0).fit(X)
        num_dice = max(clustering.labels_) + 1
        return num_dice
    else:
        return None
def count_dots2(image, redTrigger):
    params = cv2.SimpleBlobDetector_Params()

    params.minThreshold = 150 #165
    params.maxThreshold = 255
    params.blobColor = 255

    if redTrigger:
        params.blobColor = 0
        image = Segmentation.threshold(image, 150)
        image = cv2.blur(image, (3,3))
        kernel = np.ones((5,5), np.uint8)
        image = cv2.erode(image, kernel, iterations=1)
        #Utils.show(image, 'thresh')
    params.filterByArea = True
    params.minArea = 20
    params.minDistBetweenBlobs = 2
    params.filterByColor = True
    
    params.thresholdStep = 5
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(image)
    img_with_keypoints = cv2.drawKeypoints(
        image, keypoints, np.array([]), (255, 0, 0),
        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
   
    
    return keypoints, img_with_keypoints

def count_dots(image):
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 255
    params.filterByArea = True
    params.minArea = 1
    params.maxArea = 100

    # disable the default settings
    params.filterByInertia = False
    params.filterByConvexity = False
    params.filterByCircularity = True
    # 0.7 could be rectangular, too. 1 is round. Not set because the dots are not always round when they are damaged, for example.
    params.minCircularity = 0.
    params.maxCircularity = 3.4028234663852886e+38  # infinity.
    params.filterByConvexity = False
    params.minConvexity = 0.
    params.maxConvexity = 3.4028234663852886e+38

    params.filterByInertia = True  # a second way to find round blobs.
    params.minInertiaRatio = 0.5  # 1 is round, 0 is anywhat
    params.maxInertiaRatio = 3.4028234663852886e+38  # infinity again

    params.minThreshold = 50  # from where to start filtering the image
    params.maxThreshold = 255.0  # where to end filtering the image
    params.thresholdStep = 5  # steps to go through
    # avoid overlapping blobs. must be bigger than 0. Highly depending on image resolution!
    params.minDistBetweenBlobs = 2
    params.minRepeatability = 2

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(image)
    img_with_keypoints = cv2.drawKeypoints(
        image, keypoints, np.array([]), (255, 0, 0),
        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    return keypoints, img_with_keypoints
