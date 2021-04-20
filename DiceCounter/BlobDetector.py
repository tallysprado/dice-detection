import cv2
import numpy as np
from sklearn import cluster
from DiceCounter import Segmentation

def get_dice_from_blobs(blobs, image, eps=35, circle_area=27):
    """Função para obter dados a partir dos blobs detectados
    Esta função foi utilizada para contar os pontos e depois 
    localizar os dados por clusterização

    Args:
        blobs {integer}: coordenadas X e Y de retorno da função detect 
        image {matrix}: imagem onde será desenhado os blobs obtidos

    Returns:
        [integer, matrix]: retorna tupla com quantidade de dados e imagem desenhada com suas posições
    """
    X = []
    for b in blobs:
        position = b.pt
        if position != None:
            X.append(position)
    X = np.asarray(X)
    if len(X)>0:
        clustering = cluster.DBSCAN(eps=eps, min_samples=0).fit(X)
        num_dice = max(clustering.labels_) + 1

        for i in range(num_dice):
            X_dice = X[clustering.labels_==i]
            centroid_dice = np.mean(X_dice, axis=0)   
            position = (int(centroid_dice[0]), int(centroid_dice[1]))
            image = cv2.circle(image, position , circle_area, (255,0,0), 2)
            cv2.putText(
                image, str(len(X_dice)), position, cv2.FONT_HERSHEY_PLAIN, 2,
                (255,0,0), 2
            )
        return num_dice, image
    else:
        return None, image
   
def count_by_blobs(filtered_image, blobColor = 0):
    """Esta função será utilizada apenas em dados2.jpg
    
    Args:
        filtered_image (Matrix): Imagem com fundo branco e apenas os pontos pretos referentes aos dados

    Returns:
        keypoints: coordenadas de cada ponto da face
        img_with_keypoints: Utilizado para debugging, imagem com os contornos dos pontos desenhados
        
    """
    params = cv2.SimpleBlobDetector_Params()
    params.blobColor = blobColor
    params.minDistBetweenBlobs = 1
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(filtered_image)
    img_with_keypoints = cv2.drawKeypoints(
        filtered_image, keypoints, np.array([]), (255, 0, 0),
        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    cv2.imshow('image_with_keypoints', img_with_keypoints)
    return keypoints, img_with_keypoints

def count_dots_in_single_dice(image):
    """Função para contar pontos na face do dado
    A imagem de entrada é um dado de cada vez de dados3.png
    Esta função é utilizada apenas em get_dices_using_contours no arquivo Segmentation
    Ela retorna os blobs encontrados na face de cada dado
    Args:
        image (matrix): retângulo com o dado detectado em dados3.png
    Returns:
        keypoints (integer): coordenadas de cada dado detectado
        img_with_keypoints: imagem com os dados detectados (utilizado apenas para debbuging)
    """
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 150 #165
    params.maxThreshold = 255
    params.filterByArea = True
    params.minArea = 20
    params.minDistBetweenBlobs = 2
    params.filterByColor = True
    params.blobColor = 0
    image = Segmentation.threshold(image, 150)
    image = cv2.blur(image, (3,3))
    kernel = np.ones((5,5), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    params.thresholdStep = 5
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(image)
    img_with_keypoints = cv2.drawKeypoints(
        image, keypoints, np.array([]), (255, 0, 0),
        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    
    return keypoints, img_with_keypoints
