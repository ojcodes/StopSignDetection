#!/usr/bin/python2.7 python
#Minimum code required to retrieve the video. This also converts BGR into proper RGB. This is my script
 
import zmq
import numpy as np
import cv2
import pygame
from selfdrive.messaging import recv_one
from selfdrive.services import service_list
from common.transformations.camera import FULL_FRAME_SIZE

#gets data packets from zmq via port 8002
context = zmq.Context()
frame = context.socket(zmq.SUB)
frame.connect("tcp://127.0.0.1:%d" % (service_list['frame'].port)) #port 8002
frame.setsockopt(zmq.SUBSCRIBE, "")

# frame Display
while 1:
  
  fpkt = recv_one(frame)
  yuv_img = fpkt.frame.image
  yuv_transform = np.array(fpkt.frame.transform).reshape(3,3) 
  
  if yuv_img and len(yuv_img) == FULL_FRAME_SIZE[0] * FULL_FRAME_SIZE[1] * 3 // 2:
    yuv_np = np.frombuffer(yuv_img, dtype=np.uint8).reshape(FULL_FRAME_SIZE[1] * 3 // 2, -1)
    im=cv2.cvtColor(yuv_np, cv2.COLOR_YUV2RGB_I420) #yuv comes format comes from the sensor
    im=cv2.cvtColor(im, cv2.COLOR_BGR2RGB) #converts some funny BGR format into RGB

  cv2.imshow('car"s full frame',im)
  cv2.waitKey(1)
  

