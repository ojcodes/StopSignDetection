#This script detects stop signs based on haar cascade on videos
import time
import cv2 as cv
import zmq
import numpy as np
import pygame
from selfdrive.messaging import recv_one
from selfdrive.services import service_list
from common.transformations.camera import FULL_FRAME_SIZE
start = time.time()

size = (640, 480)
#_BB_OFFSET = 320,180
_BB_OFFSET = 0,0
_BB_SCALE = 1164/640.

_BB_TO_FULL_FRAME = np.asarray([
    [_BB_SCALE, 0., _BB_OFFSET[0]],
    [0., _BB_SCALE, _BB_OFFSET[1]],
    [0., 0.,   1.]])

#pygame.display.set_caption("Vision Stopsigns")
screen = pygame.display.set_mode(size, pygame.DOUBLEBUF)

camera_surface = pygame.surface.Surface((640, 480), 0, 24).convert()
#camera_surface = pygame.surface.Surface((1164, 640), 0, 24).convert() #oj
imgff = np.zeros((FULL_FRAME_SIZE[1], FULL_FRAME_SIZE[0], 3), dtype=np.uint8)
img = np.zeros((480, 640, 3), dtype='uint8')
#img = np.zeros((1164, 640, 3), dtype='uint8') #OJ

#Loading Haar Cascade in this case
path1 = '/home/oj/oj_scripts/archive/oj_stop.xml'
haar_face_cascade =cv.CascadeClassifier(path1)

# gets data packets from zmq via port 8002
context = zmq.Context()
frame = context.socket(zmq.SUB)
frame.connect("tcp://127.0.0.1:%d" % (service_list['frame'].port))  # port 8002
frame.setsockopt(zmq.SUBSCRIBE, "")
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