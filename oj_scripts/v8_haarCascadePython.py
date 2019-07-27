#This script detects stop signs based on haar cascade


import cv2 as cv
import matplotlib.pyplot as plt
import time

#Loading Image and its gray scale image
start = time.time()
img_path = '/home/oj/oj_scripts/archive/s2.jpg'
img = cv.imread(img_path) #0 filter condition means that it reads a grayscale image. H
#cv.imshow('img', img);cv.waitKey(0);cv.destroyAllWindows()
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) #converting as opencv expects a gray image

#Loading Haar Cascade in this case
path1 = '/home/oj/oj_scripts/archive/oj_stop.xml'
haar_face_cascade =cv.CascadeClassifier(path1)

#find stop signs
stops = haar_face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
print('Stop Signs found: ', len(stops))

#time it takes
end1 = time.time();print('time before plot (ms)',round(1000*(end1 - start), 1))

#go over list of stop signs and draw them as rectangles on original colored image
for (x, y, w, h) in stops:
  cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
#Displaying the rectangle over image
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))

#time it takes
end2 = time.time();print('time after plot (ms)',round(1000*(end2 - start), 2))
plt.show()
#cv.imshow('img', img);cv.waitKey(0);cv.destroyAllWindows()

#NOTES: When it comes to time consumption, it seems that image input impacts time the most