import cv2
import numpy as np
import imutils as util

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
cap.set(10,100)

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
            screenCNT = aprox
            print(screenCNT)
            height,width = img.shape[:2]
            res = cv2.resize(img,(2*width,2*height),interpolation=cv2.INTER_CUBIC)
            pts1 = screenCNT.reshape(4,2)

            rect = np.zeros((4, 2), dtype="float32")
            s = pts1.sum(axis=1)
            rect[0] = pts1[np.argmin(s)]
            rect[3] = pts1[np.argmax(s)]

            diff = np.diff(pts1, axis=1)
            rect[1] = pts1[np.argmin(diff)]
            rect[2] = pts1[np.argmax(diff)]

            pts2 = np.float32([[0, 0], [200, 0], [0, 300], [200, 300]])
            matrix = cv2.getPerspectiveTransform(rect,pts2)
            result = cv2.warpPerspective(img,matrix,(200,300))
            cv2.imshow(f"Perspective {rect[0][0]}",result)
            cv2.drawContours(img,[screenCNT],-1,(0,255,0),2)
    cv2.imshow("Video",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break