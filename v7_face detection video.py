#Minimum code required for Haar Cascade opencv example for video detection

import cv2 as cv
import numpy as np

#Loading Haar Cascade in this case
path1 = '/usr/local/lib/python2.7/dist-packages/cv2/data/haarcascade_frontalface_default.xml'
haar_face_cascade =cv.CascadeClassifier(path1)


cap = cv.VideoCapture(0) #capture video off the camera

while 1:
  ret, img = cap.read()
  gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) #converting as opencv expects a gray video
  #finding faces. cascade expects a gray video
  faces = haar_face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
  #go over list of faces and draw them as rectangles on original colored image
  for (x, y, w, h) in faces:
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv.imshow('video', img)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv.destroyAllWindows()