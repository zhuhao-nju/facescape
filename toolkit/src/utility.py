"""
Copyright 2020, Hao Zhu, NJU.
Utility functions.
"""

import numpy as np
import cv2, PIL.Image

# show image in Jupyter Notebook (work inside loop)
from io import BytesIO 
from IPython.display import display, Image
def show_img_arr(arr, bgr_mode = False):
    if bgr_mode is True:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    im = PIL.Image.fromarray(arr)
    bio = BytesIO()
    im.save(bio, format='png')
    display(Image(bio.getvalue(), format='png'))

# show depth array in Jupyter Notebook (work inside loop)
def show_depth_arr(depth_map):
    depth_max = np.max(depth_map)
    depth_min = np.min(depth_map)
    depth_map = (depth_map - depth_min)/(depth_max - depth_min)*255
    show_img_arr(depth_map.astype(np.uint8))

# rotate verts along y axis
def rotate_verts_y(verts, y):
    verts_mean = np.mean(verts, axis = 0)
    verts = verts - verts_mean

    angle = y*np.math.pi/180
    R = np.array([[np.cos(angle), 0, np.sin(angle)],
                  [0, 1, 0],
                  [-np.sin(angle), 0, np.cos(angle)]])
    
    verts = np.tensordot(R, verts.T, axes = 1).T + verts_mean
    return verts

# rotate verts along x axis
def rotate_verts_x(verts, x):
    verts_mean = np.mean(verts, axis = 0)
    verts = verts - verts_mean

    angle = x*np.math.pi/180
    R = np.array([[1, 0, 0],
                  [0, np.cos(angle), -np.sin(angle)],
                  [0, np.sin(angle), np.cos(angle)]])

    verts = np.tensordot(R, verts.T, axes = 1).T + verts_mean
    return verts

# rotate verts along z axis
def rotate_verts_z(verts, z):
    verts_mean = np.mean(verts, axis = 0)
    verts = verts - verts_mean

    angle = z*np.math.pi/180
    R = np.array([[np.cos(angle), -np.sin(angle), 0],
                  [np.sin(angle), np.cos(angle), 0],
                  [0, 0, 1]])

    verts = np.tensordot(R, verts.T, axes = 1).T + verts_mean
    return verts
