# This script detects stop signs based on haar cascade on a direct video file. Also, tries to find the distance of the stop sign
import time
import cv2 as cv


start = time.time()
# Loading Haar Cascade in this case
path1 = '/home/oj/oj_scripts/archive/oj_stop.xml';haar_face_cascade = cv.CascadeClassifier(path1)
path2 ='/home/oj/oj_scripts/videos/stopvid.mp4'

haar_face_cascade = cv.CascadeClassifier(path1)
end1 = time.time();print('checkpoint1 (ms)', round(1000 * (end1 - start), 1))
cap = cv.VideoCapture(path2)


# frame Display
while 1:
    start = time.time()
    ret, frame = cap.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    im2, contours, hierarchy = cv.findContours(gray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    stops = haar_face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4)
    # go over list of stop signs and draw them as rectangles on original colored image
    for (x, y, w, h) in stops:
      cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3) #zoomed in, x starts to go to right and y starts to go up
      #print("x",x,"w",w,"w-x", w-x,"y",y, "h", h, "y-h", h-y)
      #print ("h (pixels):", h) #X pixels @ ScaleFActor=1.2 and minNeighbours = 4, rectangle size=2
      #print ('Calibration - h [pixels]/12[ft]:',h/12) #where X pixels at 12ft is known distance for this 12ft videos.
      #print "1/(h/11.0)", 1/(h/11.0)
      dis= round(1*(133/(h/11.3)),1) #h/12 for 12 ft video gives 11, thus using it.
      print 'distance(ft):',dis
      #print("The number is " + str(dis))

    cv.imshow('frame',frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
