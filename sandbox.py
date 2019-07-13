#Haar Cascade opencv example. This one is for face detection in an image. Same could be applied to a video.
#whether an image or a video, it has to be converted into a grayscale
import cv2 as cv
#import matplotlib library
import matplotlib.pyplot as plt
#importing time library for speed comparisons of both classifiers
import time

#Loading Image and its gray scale image
img = cv.imread('s2.jpg') #0 filter condition means that it reads a grayscale image. H
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) #converting as opencv expects a gray image
cv.imshow('img',img);cv.waitKey(0);cv.destroyAllWindows()

#Loading Haar Cascade in this case
path1 = '/home/oj/Downloads/Stopsign_HAAR_19Stages.xml'
haar_face_cascade =cv.CascadeClassifier(path1)

#finding faces. cascade expects a gray image
faces = haar_face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5);
print('Faces found: ', len(faces))

#go over list of faces and draw them as rectangles on original colored image
for (x, y, w, h) in faces:
  cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.show()
#cv.imshow('img', img);cv.waitKey(0);cv.destroyAllWindows()
