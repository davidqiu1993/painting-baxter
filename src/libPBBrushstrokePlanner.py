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
import random

import pdb


class TBrushstrokePlanner(object):
  def __init__(self):
    super(TBrushstrokePlanner, self).__init__()

    self._OUTRANGE_PENALTY_FACTOR = 3

  def _simulateBrushstroke_LineSegment(self, target_image, cur_image, p_start, p_end, thickness):
    image_brushstroke = np.zeros(target_image.shape, np.uint8)

    brushstroke = (p_start[0], p_start[1], p_end[0], p_end[1])
    bounded_brushstroke = np.zeros(4, np.int)
    bounded_brushstroke[0] = max(0, min(target_image.shape[1], np.round(brushstroke[0])))
    bounded_brushstroke[1] = max(0, min(target_image.shape[0], np.round(brushstroke[1])))
    bounded_brushstroke[2] = max(0, min(target_image.shape[1], np.round(brushstroke[2])))
    bounded_brushstroke[3] = max(0, min(target_image.shape[0], np.round(brushstroke[3])))
    cv2.line(image_brushstroke, \
             (bounded_brushstroke[0], bounded_brushstroke[1]), \
             (bounded_brushstroke[2], bounded_brushstroke[3]), \
             255, \
             thickness=int(thickness))
    #cv2.imshow('image_brushstroke', image_brushstroke)

    next_image = cv2.bitwise_or(cur_image, image_brushstroke)
    #cv2.imshow('next_image', next_image)

    increment_image = cv2.bitwise_xor(next_image, cur_image)
    #cv2.imshow('increment_image', increment_image)

    image_increment_matched = cv2.bitwise_and(target_image, increment_image)
    res, image_increment_matched = cv2.threshold(image_increment_matched, 127, 255, cv2.THRESH_BINARY)
    image_mismatched = cv2.bitwise_and(image_brushstroke, np.full(target_image.shape, 255, np.uint8) - target_image)
    res, image_mismatched = cv2.threshold(image_mismatched, 127, 255, cv2.THRESH_BINARY)
    reward_positive = cv2.countNonZero(image_increment_matched)
    reward_negative = cv2.countNonZero(image_mismatched)
    reward = reward_positive - reward_negative * self._OUTRANGE_PENALTY_FACTOR
    cv2.imshow('image_increment_matched', image_increment_matched)
    cv2.imshow('image_mismatched', image_mismatched)
    
    return next_image, increment_image, reward

  def simulateBrushstroke(self, target_image, cur_image, brushstroke):
    """
    Simulate a brushstroke based on the current image.

    @param target_image The target image.
    @param cur_image The current image.
    @param brushstroke The brushstroke options.
    @return next_image
    @return increment_image
    @return reward
    """
    if brushstroke['type'] == 'LineSegment':
      return self._simulateBrushstroke_LineSegment(
        target_image, cur_image, 
        brushstroke['p_start'], 
        brushstroke['p_end'], 
        brushstroke['thickness'])
    else:
      assert(False)

  def _sampleToDrawPoints(self, target_image, cur_image, n_points):
    image_empty_area = np.full(cur_image.shape, 255, np.uint8) - cur_image
    image_to_draw = cv2.bitwise_and(image_empty_area, target_image)
    #cv2.imshow('image_to_draw', image_to_draw)
    
    locations = cv2.findNonZero(image_to_draw).tolist()
    sample_locations = random.sample(locations, n_points)

    res_points = []
    for sample_location in sample_locations:
      res_points.append(sample_location[0])

    return res_points

  def _generateInitBrushstroke_LineSegment(self, target_image, cur_image):
    N_INIT_POINTS = 5
    INIT_BRUSHSTROKE_LENGTH = 20
    INIT_BRUSHSTROKE_THICKNESS = 5

    init_points = self._sampleToDrawPoints(target_image, cur_image, N_INIT_POINTS)

    init_brushstrokes = []
    init_rewards = []
    best_init_brushstroke_index = None
    best_init_brushstroke_reward = None
    for i in range(len(init_points)):
      init_point = init_points[i]

      init_brushstroke = {
        'type': 'LineSegment',
        'p_start': np.array(init_point),
        'p_end': np.array(init_point) + np.array([INIT_BRUSHSTROKE_LENGTH, INIT_BRUSHSTROKE_LENGTH]),
        'thickness': INIT_BRUSHSTROKE_THICKNESS
      }
      init_brushstrokes.append(init_brushstroke)

      next_image, increment_image, reward = self.simulateBrushstroke(target_image, cur_image, init_brushstroke)
      init_rewards.append(reward)

      if best_init_brushstroke_index is None or best_init_brushstroke_reward is None:
        best_init_brushstroke_index = i
        best_init_brushstroke_reward = reward
      elif reward > best_init_brushstroke_reward:
        best_init_brushstroke_index = i
        best_init_brushstroke_reward = reward

    best_init_brushstroke = init_brushstrokes[best_init_brushstroke_index]

    return best_init_brushstroke

  def generateInitBrushstroke(self, target_image, cur_image, brushstroke_type):
    """
    Optimize a brushstroke based on the current image.

    @param target_image The target image.
    @param cur_image The current image.
    @param brushstroke_type The initial brushstroke type.
    @return brushstroke The options of a generated initial brushstroke.
    """
    if brushstroke_type == 'LineSegment':
      return self._generateInitBrushstroke_LineSegment(target_image, cur_image)
    else:
      assert(False)

  def _optimizeBrushstrokeOnce_LineSegment(self, target_image, cur_image, p_start, p_end, thickness):
    DERIVATIVE_PRECISION = 3
    LEARNING_RATE = 0.1

    brushstroke_results = []
    for i in range(5):
      brushstroke = [p_start[0], p_start[1], p_end[0], p_end[1]]
      if i > 0: brushstroke[i-1] += DERIVATIVE_PRECISION
      bounded_brushstroke = np.zeros(4, np.int)
      bounded_brushstroke[0] = max(0, min(target_image.shape[1], np.round(brushstroke[0])))
      bounded_brushstroke[1] = max(0, min(target_image.shape[0], np.round(brushstroke[1])))
      bounded_brushstroke[2] = max(0, min(target_image.shape[1], np.round(brushstroke[2])))
      bounded_brushstroke[3] = max(0, min(target_image.shape[0], np.round(brushstroke[3])))

      brushstroke_options = {
        'type': 'LineSegment',
        'p_start': (bounded_brushstroke[0], bounded_brushstroke[1]),
        'p_end': (bounded_brushstroke[2], bounded_brushstroke[3]),
        'thickness': thickness
      }

      next_image, increment_image, reward = self.simulateBrushstroke(target_image, cur_image, brushstroke_options)

      brushstroke_results.append((brushstroke_options, next_image, increment_image, reward))

    reward_derivatives = []
    reward_directions = []
    for i in range(4):
      brushstroke_options_0, next_image_0, increment_image_0, reward_0 = brushstroke_results[0]
      brushstroke_options_i, next_image_i, increment_image_i, reward_i = brushstroke_results[i+1]
      reward_derivative = float(reward_i - reward_0) / float(DERIVATIVE_PRECISION)
      if reward_i - reward_0 == 0:
        reward_direction = 0
      else:
        reward_direction = (reward_i - reward_0) / abs(reward_i - reward_0)
      reward_derivatives.append(reward_derivative)
      reward_directions.append(reward_direction)

    brushstroke = np.array([p_start[0], p_start[1], p_end[0], p_end[1]])
    #brushstroke = brushstroke + np.round(np.array(reward_directions)) * DERIVATIVE_PRECISION
    brushstroke = brushstroke + np.array(reward_derivatives) * DERIVATIVE_PRECISION * LEARNING_RATE
    bounded_brushstroke = np.zeros(4, np.int)
    bounded_brushstroke[0] = max(0, min(target_image.shape[1], np.round(brushstroke[0])))
    bounded_brushstroke[1] = max(0, min(target_image.shape[0], np.round(brushstroke[1])))
    bounded_brushstroke[2] = max(0, min(target_image.shape[1], np.round(brushstroke[2])))
    bounded_brushstroke[3] = max(0, min(target_image.shape[0], np.round(brushstroke[3])))
    brushstroke_options = {
      'type': 'LineSegment',
      'p_start': (bounded_brushstroke[0], bounded_brushstroke[1]),
      'p_end': (bounded_brushstroke[2], bounded_brushstroke[3]),
      'thickness': thickness
    }

    return brushstroke_options

  def _optimizeBrushstroke_LineSegment(self, target_image, cur_image, p_start, p_end, thickness):
    MAX_NO_IMPROVEMENT_COUNT = 20

    max_reward = None
    no_improvement_count = 0
    for i in range(1000):
      brushstroke_options = self._optimizeBrushstrokeOnce_LineSegment(target_image, cur_image, p_start, p_end, thickness)
      p_start = brushstroke_options['p_start']
      p_end = brushstroke_options['p_end']
      
      next_image, increment_image, reward = self.simulateBrushstroke(target_image, cur_image, brushstroke_options)
      if max_reward is None:
        max_reward = reward
        no_improvement_count = 0
      elif reward > max_reward:
        max_reward = reward
        no_improvement_count = 0
      else:
        no_improvement_count += 1

      print('reward >>> {} [{}]'.format(reward, no_improvement_count))
      cv2.imshow('next_image', next_image)
      cv2.waitKey(50)

      if no_improvement_count > MAX_NO_IMPROVEMENT_COUNT:
        break;

    return brushstroke_options, next_image, increment_image, reward

  def optimizeBrushstroke(self, target_image, cur_image, brushstroke):
    """
    Optimize a brushstroke based on the current image.

    @param target_image The target image.
    @param cur_image The current image.
    @param brushstroke The initial brushstroke options.
    @return brushstroke
    @return next_image
    @return increment_image
    @return reward
    """
    if brushstroke['type'] == 'LineSegment':
      return self._optimizeBrushstroke_LineSegment(
        target_image, cur_image,
        brushstroke['p_start'], 
        brushstroke['p_end'], 
        brushstroke['thickness'])
    else:
      assert(False)

  def generateNextBrushstroke(self, target_image, cur_image, brushstroke_type):
    """
    Generate the next optimized brushstroke based on the current image.

    @param target_image The target image.
    @param cur_image The current image.
    @param brushstroke_type The brushstroke type.
    @return (brushstroke, next_image, increment_image, reward)
    """
    init_brushstroke = self.generateInitBrushstroke(target_image, cur_image, brushstroke_type)
    optimized_brushstroke, next_image, increment_image, reward = self.optimizeBrushstroke(target_image, cur_image, init_brushstroke)

    return optimized_brushstroke, next_image, increment_image, reward


if __name__ == '__main__':
  target_image_raw = cv2.imread('../data/008.png')
  target_image = cv2.cvtColor(target_image_raw, cv2.COLOR_BGR2GRAY)
  cur_image = np.zeros(target_image.shape, np.uint8)
  
  planner = TBrushstrokePlanner()

  acc_reward = 0
  for i in range(100):
    brushstroke_options, next_image, increment_image, reward = planner.generateNextBrushstroke(target_image, cur_image, 'LineSegment')
    if reward > 0:
      cur_image = next_image
      acc_reward += reward

  print('acc_reward = {}'.format(acc_reward))
  cv2.imshow('final', cur_image)
  cv2.imshow('reference', target_image)
  cv2.waitKey()


