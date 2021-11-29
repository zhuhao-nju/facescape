import cv2
from src.facescape_fitter import facescape_fitter
import numpy as np
from src.facescape_bm import facescape_bm
import timeit

np.random.seed(1000)

model = facescape_bm("./bilinear_model_v1.6/facescape_bm_v1.6_847_50_52_id_front.npz")

fs_fitter = facescape_fitter(fs_file = "./bilinear_model_v1.6/facescape_bm_v1.6_847_50_52_id_front.npz",
                             kp2d_backend = 'dlib') # or 'face_alignment'

src_img = cv2.imread("./test_data/chan.jpg")

starttime = timeit.default_timer()
print("The start time is :",starttime)
# Fit 3DMM parameters
print('Fitting 3DMM parameters')
kp2d = fs_fitter.detect_kp2d(src_img) # extract 2D key points
mesh, params = fs_fitter.fit_kp2d(kp2d, model) # fit model
id, exp, scale, trans, rot_vector = params
print("Fit params: the time difference is :", timeit.default_timer() - starttime)
id = np.random.normal(model.id_mean, np.sqrt(model.id_var))
random_color_vec = (np.random.random(100)) * 100

# Warp texture
print('Warping texture')
starttime = timeit.default_timer()
#verts_img = fs_fitter.project(mesh.vertices, rot_vector, scale, trans, keepz=False)
verts_img = fs_fitter.project(mesh.vertices, np.array([0, 0, 0], dtype=np.double), 1., np.array([0, 0]), keepz=False)
texture = fs_fitter.get_texture(src_img, verts_img, mesh, model)
print("Warped tex: the time difference is :", timeit.default_timer() - starttime)

cv2.imwrite("./demo_output/texture.jpg", texture)

# Reading expression from pictures
exp_img = cv2.imread("./test_data/expression4.png")
kp2d = fs_fitter.detect_kp2d(exp_img) # extract 2D key points
_, params = fs_fitter.fit_kp2d(kp2d, model) # fit model
_, exp_vec, _, _, _ = params
# # # create random expression vector
# exp_vec = np.zeros(52)
# exp_vec[np.random.randint(52)] = 1
# exp_vec[20] = 1
exp_vec *= 100
starttime = timeit.default_timer()
# generate and save full head mesh
mesh_full = model.gen_full(id, exp_vec)
print("Animating model:", timeit.default_timer() - starttime)

mesh_full.export(output_dir="./demo_output/", file_name="bm_v16_result_full_matthijs_inferred_exp", texture_name="texture.jpg")

# Create exp vec
exp_vec = np.zeros(52)
exp_vec[0] = 0.5
exp_vec[1] = 0
exp_vec[2] = 0
exp_vec[20] = 0
exp_vec[31] = 10
exp_vec[32] = 10


mesh_full = model.gen_full(id, exp_vec)
mesh_full.export(output_dir="./demo_output/", file_name="bm_v16_result_full_matthijs_handcrafted_exp", texture_name="texture.jpg")

# # generate and save facial mesh
# mesh_face = model.gen_face(id, exp_vec)
# mesh_face.export(output_dir="./demo_output/", file_name="bm_v16_result_face_matthijs", texture_name="texture.jpg")
#
# # generate and save facial mesh with rough vertex color
# mesh_face_color = model.gen_face_color(id, exp_vec, random_color_vec)
# mesh_face_color.export("./demo_output/bm_v16_result_face_color_matthijs.obj", enable_vc=True)
#
# print("Results saved to './demo_output/'")