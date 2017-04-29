#!/usr/bin/python

"""
Painting Baxter outline planner libaray.
"""

__author__    = 'David Qiu'
__email__     = 'david@davidqiu.com'
__website__   = 'www.davidqiu.com'
__copyright__ = 'Copyright (C) 2017, David Qiu. All rights reserved.'


import libPBUtils as utils

import cv2
import numpy as np

import pdb


class TOutlinePlanner(object):
  def __init__(self):
    super(TOutlinePlanner, self).__init__()

  def find_outlines(self, target_image, sparsity=1):
    ret, image_binary = cv2.threshold(target_image, 15, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5,5), np.uint8) # erosion kernal
    erosion = cv2.erode(image_binary, kernel, iterations = 1)

    opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel) # denoise
    #cv2.imshow('closing', closing)

    image_contours_tmp, contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    filtered_contours = []
    outlines = []
    for i_contour in range(len(contours)):
      contour = contours[i_contour]
      filtered_contours.append([])
      outlines.append([])
      for i_point in range(len(contour)):
        if i_point % sparsity == 0:
          point_x = contour[i_point][0][0]
          point_y = contour[i_point][0][1]
          filtered_contours[i_contour].append([[point_x, point_y]])
          outlines[i_contour].append((point_x, point_y))
      filtered_contours[i_contour] = np.array(filtered_contours[i_contour])

    return outlines, filtered_contours


if __name__ == '__main__':
  target_image_raw = cv2.imread('../data/009.jpg')
  target_image = cv2.cvtColor(target_image_raw, cv2.COLOR_BGR2GRAY)
  cv2.imshow('reference', target_image)

  cmd_file = 'draw.cmd'
  utils.flushcmd(cmd_file)

  planner = TOutlinePlanner()

  outlines, contours = planner.find_outlines(target_image, sparsity=1)

  for i_outline in range(len(outlines)):
    print('OUTLINE #{}'.format(i_outline))
    utils.outcmd(cmd_file, str('outline'))
    for i_point in range(len(outlines[i_outline])):
      point_x, point_y = outlines[i_outline][i_point]
      norm_px, norm_py = utils.norm_point(point_x, point_y, target_image.shape[0], target_image.shape[1])
      print('({}, {})'.format(point_x, point_y))
      utils.outcmd(cmd_file, '{} {}'.format(norm_px, norm_py))

  image_contours = np.zeros(target_image.shape)
  cv2.drawContours(image_contours, contours, -1, (255, 255, 255), 4)
  cv2.imshow('image_contours', image_contours)
  cv2.waitKey()


