import cv2
from src.facescape_fitter import facescape_fitter
import numpy as np
from src.facescape_bm import facescape_bm
import timeit
#import cupy as cp
import csv
import numpy as np, cv2, trimesh
from src.facescape_fitter import facescape_fitter
from src.renderer import render_orthcam

np.random.seed(1000)


def read_open_face_expressions(file='./test_data/src.csv'):
    """
    
    
    :param file: 
    :return: 
    """

    expressions = {}
    with open(file) as f:
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
    frame_count = int(expressions['frame'][-1])
    exp_vec = np.zeros((52, frame_count))

    # # # Setting heuristic expressions
    # exp_vec[9, :] += 1  # EyeOpen_L
    # exp_vec[10, :] += 1  # EyeOpen_R

    # Reading expressions from OpenFace data source file
    exp_vec[1, :] = np.array(expressions['AU45_r'])
    exp_vec[2, :] = np.array(expressions['AU45_r'])
    exp_vec[3, :] = np.array(expressions['AU07_r'])
    exp_vec[4, :] = np.array(expressions['AU07_r'])
    exp_vec[13, :] = np.array(expressions['AU05_r'])
    exp_vec[14, :] = np.array(expressions['AU05_r'])
    exp_vec[15, :] = np.array(expressions['AU04_r'])
    exp_vec[16, :] = np.array(expressions['AU04_r'])
    exp_vec[17, :] = np.array(expressions['AU01_r'])
    exp_vec[18, :] = np.array(expressions['AU02_r'])
    exp_vec[19, :] = np.array(expressions['AU02_r'])
    exp_vec[20, :] = np.array(expressions['AU26_r'])
    exp_vec[21, :] = np.array(expressions['AU28_c'])
    exp_vec[25, :] = np.array(expressions['AU10_r'])
    exp_vec[26, :] = np.array(expressions['AU10_r'])
    exp_vec[27, :] = np.array(expressions['AU15_r'])
    exp_vec[28, :] = np.array(expressions['AU15_r'])
    exp_vec[33, :] = np.array(expressions['AU14_r'])
    exp_vec[34, :] = np.array(expressions['AU14_r'])
    exp_vec[35, :] = np.array(expressions['AU12_r'])
    exp_vec[36, :] = np.array(expressions['AU12_r'])
    exp_vec[41, :] = np.array(expressions['AU23_r'])
    exp_vec[42, :] = np.array(expressions['AU25_r'])
    exp_vec[43, :] = np.array(expressions['AU20_r'])
    exp_vec[44, :] = np.array(expressions['AU20_r'])
    exp_vec[46, :] = np.array(expressions['AU17_r'])
    exp_vec[50, :] = np.array(expressions['AU06_r'])
    exp_vec[51, :] = np.array(expressions['AU06_r'])

    exp_vec *= 0.4

    # Heuristically setting eye related vecs
    exp_vec[3, :] *= 0
    exp_vec[4, :] *= 0
    exp_vec[5, :] *= 0
    exp_vec[6, :] *= 0
    exp_vec[7, :] *= 0
    exp_vec[8, :] *= 0
    exp_vec[9, :] *= 0
    exp_vec[10, :] *= 0
    exp_vec[11, :] *= 0
    exp_vec[12, :] *= 0
    exp_vec[13, :] *= 0
    exp_vec[14, :] *= 0

    # Opening eyes manually
    exp_vec[9, :] += 2.5
    exp_vec[10, :] += 2.5

    return exp_vec


# Initialize model and fitter
model = facescape_bm("./bilinear_model_v1.6/facescape_bm_v1.6_847_50_52_id_front.npz")
fs_fitter = facescape_fitter(fs_file = "./bilinear_model_v1.6/facescape_bm_v1.6_847_50_52_id_front.npz",
                             kp2d_backend = 'dlib') # or 'face_alignment'

# Generate generic, id-mean, model
id = (np.random.random(50) - 0.5) * 0.1
if id[0]>0:
    id = -id

src_img = cv2.imread("./test_data/expression.png")
# kp2d = fs_fitter.detect_kp2d(src_img)  # extract 2D key points
# _, params = fs_fitter.fit_kp2d(kp2d, model)  # fit model
# id, exp, scale, trans, rot_vector = params

# Read expressions from openFace recording
exp_vecs = read_open_face_expressions()
frame_count = exp_vecs.shape[1]

# Generate al 52 blendshape meshes
# blendshape_meshes = []
# for i in range(52):
#     print(f'Starting blendshape {i}')
#     exp_vec = np.zeros(52)
#     #exp_vec[0] = 1
#     exp_vec[i] = 3
#     verts = model.gen_full(id, exp_vec)
#
#     if i == 0:
#         blendshape_meshes.append(verts)
#     else:
#         blendshape_meshes.append(verts - blendshape_meshes[0])
#     # transform to orthogonal camera coordinate
#     mesh_tm = trimesh.Trimesh(vertices=verts.copy(),
#                               faces=fs_fitter.fv_indices_front - 1,
#                               process=False)
#     mesh_tm.vertices[:, :2] = mesh_tm.vertices[:, 0:2] - np.array([src_img.shape[1] / 2, src_img.shape[0] / 2])
#     mesh_tm.vertices = mesh_tm.vertices / src_img.shape[0] * 2
#
#     mesh_tm.vertices[:, 0] -= mesh_tm.vertices[:, 0].mean()
#     mesh_tm.vertices[:, 1] -= mesh_tm.vertices[:, 1].mean()
#     mesh_tm.vertices[:, 2] = mesh_tm.vertices[:, 2] - mesh_tm.vertices[:, 2].mean() - 10.
#
#     # render texture image and depth
#     rend_depth, rend_tex = render_orthcam(mesh_tm, (1, 1),
#                                           rend_size=tuple(src_img.shape[:2]),
#                                           flat_shading=False)
#     cv2.imwrite(f'./demo_output/blendshapes/{i}.png', rend_tex)

w_0 = np.zeros(52)
w_0[0] = 1

# # Animate blendshape meshes based on OpenFace recorded expression vectors
for i in range(frame_count):
    w = exp_vecs[:, i]

    w_exp = (w + (1 - np.sum(w))*w_0)
    verts = model.gen_full(id, w_exp)

    # transform to orthogonal camera coordinate
    mesh_tm = trimesh.Trimesh(vertices=verts.copy(),
                              faces=fs_fitter.fv_indices_front - 1,
                              process=False)
    mesh_tm.vertices[:, :2] = mesh_tm.vertices[:, 0:2] - np.array([src_img.shape[1] / 2, src_img.shape[0] / 2])
    mesh_tm.vertices = mesh_tm.vertices / src_img.shape[0] * 2

    mesh_tm.vertices[:, 0] -= mesh_tm.vertices[:, 0].mean()
    mesh_tm.vertices[:, 1] -= mesh_tm.vertices[:, 1].mean()
    mesh_tm.vertices[:, 2] -= mesh_tm.vertices[:, 2].mean() + 1.

    # render texture image and depth
    rend_depth, rend_tex = render_orthcam(mesh_tm, (1, 1),
                                      rend_size = tuple(src_img.shape[:2]),
                                      flat_shading=False)
    cv2.imwrite(f'./demo_output/{i}.png', rend_tex)
    print(f'Rendered frame {i}')
