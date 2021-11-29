import cv2
from src.facescape_fitter import facescape_fitter
import numpy as np
from src.facescape_bm import facescape_bm
import timeit
import cupy as cp
import csv

np.random.seed(1000)

## Read expression vec from file
expression_file = './toolkit/test_data/src.csv'
expressions = {}
with open(expression_file) as f:
    records = csv.DictReader(f)
    for row in records:
         for key in row.keys():
            # strip first whitespace from keys
            expkey = key[1:] if key[0] == ' ' else key

            if expkey in expressions.keys():
                expressions[expkey].append(float(row[key]))
            else:
                expressions[expkey] = [float(row[key])]

# Map expressions from OpenFace AU to FaceScape


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
id_cp = cp.array(id)
exp_cp = cp.array(exp_vec)
mesh = model.gen_full_cupy(id_cp, exp_cp)
print("Animating model cupy:", timeit.default_timer() - starttime)
#mesh.export(output_dir="./demo_output/", file_name="bm_v16_result_full_matthijs_inferred_exp")

for i in range(10):
    start_frame = timeit.default_timer()
    starttime = timeit.default_timer()
    exp_cp += 0.5
    verts = model.gen_full_cupy(id_cp, exp_cp)
    print(f"Animating model cupy {i}", timeit.default_timer() - starttime)
    #mesh.export(output_dir="./demo_output/", file_name="bm_v16_result_full_matthijs_inferred_exp")
    verts_np = cp.asnumpy(verts)
    print(f"total time: {i}", timeit.default_timer() - start_frame)


# # Generate and save animated head mesh
# starttime = timeit.default_timer()
# mesh = model.gen_full(id, exp_vec)
# print("Animating model:", timeit.default_timer() - starttime)
