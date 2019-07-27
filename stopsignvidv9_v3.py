# This script detects stop signs based on haar cascade on a direct video file. Also, tries to find the distance of the stop sign
import time
import cv2 as cv


start = time.time()
# Loading Haar Cascade in this case
path1 = '/home/oj/oj_scripts/archive/oj_stop.xml';haar_face_cascade = cv.CascadeClassifier(path1)
path2 ='/home/oj/oj_scripts/videos/12ft.mp4'

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
      cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
      #print("x",x,"w",w,"w-x", w-x,"y",y, "h", h, "y-h", h-y)
      #print ("h (pixels):", h) #h=w=~135 pixels
      #print ('Calibration - h [pixels]/12[ft]:',h/12) #where h pixels at 12ft. 12 ft was known from video "12ft"
      #print "1/(h/11.0)", 1/(h/11.0) #h/12 gives~11, so divide h/11 but since more pixels means less distance so using 1/(h/11)
      dis= round(133/(h/11.3),1)  #1/(h/11) gives roughly 0.08 so 12/(1/(h/11)) gives a factor of ~140. Using it for scaling
      print 'distance(ft):',dis
      

    cv.imshow('frame',frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
