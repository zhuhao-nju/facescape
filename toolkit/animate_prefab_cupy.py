import cv2
from src.facescape_fitter import facescape_fitter
import numpy as np
from src.facescape_bm import facescape_bm
import timeit

np.random.seed(1000)


# Initialize model and fitter
model = facescape_bm("./bilinear_model_v1.6/facescape_bm_v1.6_847_50_52_id_front.npz")
fs_fitter = facescape_fitter(fs_file = "./bilinear_model_v1.6/facescape_bm_v1.6_847_50_52_id_front.npz",
                             kp2d_backend = 'dlib') # or 'face_alignment'

# Generate generic, id-mean, model
# id = np.random.normal(model.id_mean, np.sqrt(model.id_var))

id = (np.random.random(50) - 0.5) * 0.1
if id[0]>0:
    id = -id

src_img = cv2.imread("./test_data/expression.png")
kp2d = fs_fitter.detect_kp2d(src_img)  # extract 2D key points
_, params = fs_fitter.fit_kp2d(kp2d, model)  # fit model
id, exp, scale, trans, rot_vector = params

# TODO: generate texture

# Read expression from pictures
# exp_img = cv2.imread("./test_data/expression4.png")
# kp2d = fs_fitter.detect_kp2d(exp_img) # extract 2D key points
# exp_mesh, exp_params = fs_fitter.fit_kp2d(kp2d, model) # fit model
# _, exp_vec, _, _, _ = exp_params

# create random expression vector
exp_vec = np.zeros(52)
exp_vec[21] = 1

# Generate and save animated head mesh
starttime = timeit.default_timer()
mesh = model.gen_full(id, exp_vec)
print("Animating model:", timeit.default_timer() - starttime)
mesh.export(output_dir="./demo_output/", file_name="bm_v16_result_full_matthijs_inferred_exp")
