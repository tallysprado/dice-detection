import cv2
import numpy as np
from DiceCounter import BlobDetector

def isRedImage(image):
    """Função auxiliar para definir se o método de segmentação será 
    contar os dados a partir dos blobs ou a partir dos contornos

    Args:
        image (matrix): imagem original sem filtros

    Returns:
        True: se o valor mediano de pixels vermelhos for acima de 200
        False: se o valor da intensidade dos pixels vermelhos for abaixo de 200
    """
    r, g, b = cv2.split(image)
    if np.median(r)>200:
        return True
    else:
        return False

def threshold(img, value):
    out = img.copy()
    ret, out = cv2.threshold(out, value, 255, cv2.THRESH_BINARY_INV)
    
    return out

def applyFilters(image, erode_kernel=4, min_threshold=220):
    """Esta função será utilizada apenas na contagem de dados a partir dos pontos (blobs)
    Os filtros minimizarão todos os ruídos e maximizarão os pontos necessários
    Args:
        image (matrix): imagem original 
        erode_kernel (int, optional): [description]. Defaults to 4.
        min_threshold (int, optional): [description]. Defaults to 220.

    Returns:
        Matrix: imagem filtrada sem ruídos e pontos maximizados
    """
    image = cv2.blur(image, (3,3))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = threshold(image, min_threshold)
    kernel = np.ones((erode_kernel,erode_kernel), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    return image

def get_dices_using_contours(image):    
    """Esta função será utilizada apenas em dados3
    Na localização dos dados a partir dos contornos
    Args:
        image (matrix): dados3

    Returns:
        Matrix: imagem com os dados localizados e contados
    """
    canny = cv2.Canny(image, 100, 550)
    out = image.copy()
    cnts, hier = cv2.findContours(
        canny.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w > 40:
            new_img = out[y:y+h, x:x+w]
            r, g, b = cv2.split(new_img)
            blurred = cv2.blur(r, (3,3))
            dots, img = BlobDetector.count_dots_in_single_dice(blurred)
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeft = (x,y)
            fontScale = 1
            fontColor = (255,0,0)
            lineType = 2
            cv2.putText(
                out, str(len(dots)), bottomLeft, font, fontScale, fontColor, lineType
            )      
            cv2.rectangle(out, (x, y), (x+w, y+h), (255, 0, 0), 1)
    return out

def get_dices(image):
    """Esta função apenas decide qual método de detecção será usado
    Se irá separar os dados para serem detectados por imagem: isRedImage true
    Se irá detectar os dados a partir dos pontos: isRedImage false
    
    isRedImage é o ruído da terceira imagem que apresenta um plano de fundo com tonalidade branca

    Args:
        image (matrix): imagem original
    """
    if isRedImage(image):
        dices = get_dices_using_contours(image)
        cv2.imshow('dados3.png',dices)
    else:
        dices = applyFilters(image)
        keypoints, image_blobs = BlobDetector.count_by_blobs(dices)
        num, marked_image = BlobDetector.get_dice_from_blobs(
            keypoints, image)
        cv2.imshow('dados2.jpg', marked_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
        