# This script detects stop signs based on haar cascade on a direct video file. Smallest code
import time
import cv2 as cv

start = time.time()
# Loading Haar Cascade in this case
path1 = '/home/oj/oj_scripts/archive/oj_stop.xml';haar_face_cascade = cv.CascadeClassifier(path1)

end1 = time.time();print('checkpoint1 (ms)', round(1000 * (end1 - start), 1))
cap = cv.VideoCapture('stopvid1.mp4')
# frame Display
while 1:
    start = time.time()
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    stops = haar_face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4)
    # go over list of stop signs and draw them as rectangles on original colored image
    for (x, y, w, h) in stops:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv.imshow('frame', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    end2 = time.time();print('checkpoint2 (ms)', round(1000 * (end2 - start), 1))

cap.release()
cv2.destroyAllWindows()