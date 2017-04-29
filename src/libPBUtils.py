#!/usr/bin/python

"""
Painting Baxter common utilities libaray.
"""

__author__    = 'David Qiu'
__email__     = 'david@davidqiu.com'
__website__   = 'www.davidqiu.com'
__copyright__ = 'Copyright (C) 2017, David Qiu. All rights reserved.'


import numpy as np


def norm_point(x, y, width, height):
  norm_x = (float(x)/float(width))
  norm_y = (float(y)/float(height))
  return norm_x, norm_y


def outcmd(cmd_file, cmd_line):
  with open(cmd_file, 'a') as f:
    f.write(cmd_line + '\n')


def flushcmd(cmd_file):
  with open(cmd_file, 'w') as f:
    f.write('')


