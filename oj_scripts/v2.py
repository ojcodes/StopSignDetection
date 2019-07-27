#!/usr/bin/python2.7 python
#This is Ethans script. It replays the 640x480 video. Has some unwanted code. 

import zmq
import numpy as np
import cv2
import pygame
from selfdrive.messaging import recv_one
from selfdrive.services import service_list
from common.transformations.camera import FULL_FRAME_SIZE

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

context = zmq.Context()
frame = context.socket(zmq.SUB)
frame.connect("tcp://127.0.0.1:%d" % (service_list['frame'].port)) #port 8002
frame.setsockopt(zmq.SUBSCRIBE, "")

# ***** frame *****
counter=0

while 1:
  list(pygame.event.get())
  fpkt = recv_one(frame)
  yuv_img = fpkt.frame.image
  yuv_transform = np.array(fpkt.frame.transform).reshape(3,3)

  if yuv_img and len(yuv_img) == FULL_FRAME_SIZE[0] * FULL_FRAME_SIZE[1] * 3 // 2:
    yuv_np = np.frombuffer(yuv_img, dtype=np.uint8).reshape(FULL_FRAME_SIZE[1] * 3 // 2, -1)
    cv2.cvtColor(yuv_np, cv2.COLOR_YUV2RGB_I420, dst=imgff)
  cv2.warpAffine(imgff, np.dot(yuv_transform, _BB_TO_FULL_FRAME)[:2], (img.shape[1], img.shape[0]), dst=img, flags=cv2.WARP_INVERSE_MAP) #THis makes actual screen
  pygame.surfarray.blit_array(camera_surface, img.swapaxes(0,1))
  screen.blit(camera_surface, (0, 0))
  pygame.display.flip()

#show image

  #cv2.imshow('hey',imgff)     
  #cv2.waitKey(1)
#
