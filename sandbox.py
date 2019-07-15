#This script detects stop signs based on haar cascade on videos
import time
import cv2 as cv
import matplotlib.pyplot as plt
import zmq
import numpy as np
import pygame
from selfdrive.messaging import recv_one
from selfdrive.services import service_list
from common.transformations.camera import FULL_FRAME_SIZE
start = time.time()

#Loading Haar Cascade in this case
path1 = '/home/oj/oj_scripts/archive/oj_stop.xml'
haar_face_cascade =cv.CascadeClassifier(path1)

end1 = time.time();print('checkpoint1 (ms)',round(1000*(end1 - start), 1))

# gets data packets from zmq via port 8002
context = zmq.Context()
frame = context.socket(zmq.SUB)
frame.connect("tcp://127.0.0.1:%d" % (service_list['frame'].port))  # port 8002
frame.setsockopt(zmq.SUBSCRIBE, "")

end2 = time.time();print('checkpoint2 (ms)',round(1000*(end1 - start), 1))
end2 = time.time();print('checkpoint2 (ms)',round(1000*(end1 - start), 1))
# frame Display
while 1:

  fpkt = recv_one(frame)
  yuv_img = fpkt.frame.image
  yuv_transform = np.array(fpkt.frame.transform).reshape(3, 3)
  end3 = time.time();print('checkpoint3 (ms)', round(1000 * (end1 - start), 1))

  if yuv_img and len(yuv_img) == FULL_FRAME_SIZE[0] * FULL_FRAME_SIZE[1] * 3 // 2:
    yuv_np = np.frombuffer(yuv_img, dtype=np.uint8).reshape(FULL_FRAME_SIZE[1] * 3 // 2, -1)
    im = cv.cvtColor(yuv_np, cv.COLOR_YUV2BGRA_I420)  # yuv comes format comes from the sensor
    gray = cv.cvtColor(im, cv.COLOR_RGB2GRAY)  # yuv comes format comes from the sensor
    #gray = cv.cvtColor(im, cv.COLOR_RGB2GRAY)  # converts some funny BGR format into RGB
    end4 = time.time();print('checkpoint4 (ms)', round(1000 * (end1 - start), 1))

    stops = haar_face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4)
    end5 = time.time();print('checkpoint5 (ms)', round(1000 * (end1 - start), 1))

    # go over list of stop signs and draw them as rectangles on original colored image
    for (x, y, w, h) in stops:
      cv.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
      end6 = time.time();print('checkpoint6 (ms)', round(1000 * (end1 - start), 1))
  cv.imshow('car"s full frame', im)
  cv.waitKey(1)

# Displaying the rectangle over image
  #plt.imshow(cv.cvtColor(im, cv.COLOR_BGR2RGB))
  #plt.show()
  #cv.imshow('cars full frame', im)