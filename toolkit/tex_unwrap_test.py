import cv2
from src.facescape_fitter import facescape_fitter
import numpy as np
from src.facescape_bm import facescape_bm
from src.renderer import render_cvcam
import timeit
# import cupy as cp
import csv
import numpy as np, cv2, trimesh
from src.facescape_fitter import facescape_fitter
from src.renderer import render_orthcam
from src.renderer import render_cvcam


np.random.seed(1000)

# Initialize model and fitter
fs_fitter = facescape_fitter(fs_file="./bilinear_model_v1.6/facescape_bm_v1.6_847_50_52_id_front.npz",
                             kp2d_backend='dlib')  # or 'face_alignment'

# Fit id to image
src_path = "./test_data/chan.jpg"
src_img = cv2.imread(src_path)
kp2d = fs_fitter.detect_kp2d(src_img)  # extract 2D key points
mesh, params, mesh_verts_img = fs_fitter.fit_kp2d(kp2d)  # fit model
id, _, scale, trans, rot_vector = params

# Get texture
texture = fs_fitter.get_texture(src_img, mesh_verts_img, mesh)
filename = './demo_output/test_mesh.jpg'
cv2.imwrite(filename, texture)

# Save base mesh
mesh.export(output_dir='./demo_output', file_name='test_mesh', texture_name='test_mesh.jpg', enable_vc=False, enable_vt=True)
