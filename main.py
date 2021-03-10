import cv2
import numpy as np
import imutils as util

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
cap.set(10,100)

def extractCard(aprox,img):
    pts1 = aprox.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts1.sum(axis=1)
    rect[0] = pts1[np.argmin(s)]
    rect[3] = pts1[np.argmax(s)]

    diff = np.diff(pts1, axis=1)
    rect[1] = pts1[np.argmin(diff)]
    rect[2] = pts1[np.argmax(diff)]

    pts2 = np.float32([[0, 0], [200, 0], [0, 300], [200, 300]])
    matrix = cv2.getPerspectiveTransform(rect, pts2)
    result = cv2.warpPerspective(img, matrix, (200, 300))
    return result

def extractCornor(card):

    cornor = card[0:70,1:25]

    cornor = cv2.cvtColor(cornor,cv2.COLOR_RGB2GRAY)

    _, cornor = cv2.threshold(cornor, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    return cornor

def saveCornor(cornor):
    card = input("Card ID: ")
    cv2.imwrite(f"Resources/{card}.png",cornor)

h4 = cv2.imread("Resources/1.png")
h4 = cv2.resize(h4,(70,24))

while True:
    success, img = cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    squere = cv2.Canny(gray,150,150)
    k = np.ones([2,2],np.uint8)
    squere = cv2.dilate(squere,k,iterations=2)

    see = cv2.findContours(squere.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    see = util.grab_contours(see)
    see = sorted(see,key=cv2.contourArea,reverse=True)[:3]

    for s in see:
        lnked = cv2.arcLength(s,True)
        aprox = cv2.approxPolyDP(s,0.02*lnked,True)

        if len(aprox) == 4:

            result = extractCard(aprox,img)
            cornor = extractCornor(result)

            score, diff = from

            cv2.imshow("Cornor", cornor)
            #saveCornor(cornor)
            cv2.imshow("Perspective",result)
            cv2.drawContours(img,[aprox],-1,(0,255,0),2)

    cv2.imshow("Video",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break