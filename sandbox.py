# This script detects stop signs based on haar cascade on a direct video file
import time
import cv2 as cv
import numpy as np

start = time.time()
# Loading Haar Cascade in this case
path1 = '/home/oj/oj_scripts/archive/oj_stop.xml'
haar_face_cascade = cv.CascadeClassifier(path1)

end1 = time.time();print('checkpoint1 (ms)', round(1000 * (end1 - start), 1))
cap = cv.VideoCapture('stopvidtrim1.mp4')
# frame Display
while 1:
    start = time.time()
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    #for red sing
    # converting from BGR to HSV color space
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Range for lower red
    lower_red = np.array([0, 100, 120])
    upper_red = np.array([10, 255, 255])
    mask1 = cv.inRange(hsv, lower_red, upper_red)
    """
    # Range for upper range
    lower_red = np.array([170, 75, 70])
    upper_red = np.array([180, 255, 255])
    mask2 = cv.inRange(hsv, lower_red, upper_red)
    """
    # Generating the final mask to detect red color
    #mask = mask1 + mask2
    mask=mask1
    # Bitwise-AND mask and original image
    res = cv.bitwise_and(frame, frame, mask=mask)
    im2, contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    cv.imshow('mask', mask)
    #cv.imshow('res', res)

    # Generating the final
    stops = haar_face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4)
    # go over list of stop signs and draw them as rectangles on original colored image
    for (x, y, w, h) in stops:
      cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
      cv.drawContours(frame, contours, -1, (0, 255, 0), 3)
    cv.imshow('frame', frame)
    if cv.waitKey(100) & 0xFF == ord('q'):
        break
    end2 = time.time();
    print('checkpoint2 (ms)', round(1000 * (end2 - start), 1))

cap.release()
cv.destroyAllWindows()