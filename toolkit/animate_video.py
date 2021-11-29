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
    exp_vec[9, :] += 1
    exp_vec[10, :] += 1

    # The expression are relative to a neutral expression
    exp_vec[0, :] = 1

    return exp_vec


# Initialize model and fitter
model = facescape_bm("./bilinear_model_v1.6/facescape_bm_v1.6_847_50_52_id_front.npz")
fs_fitter = facescape_fitter(fs_file="./bilinear_model_v1.6/facescape_bm_v1.6_847_50_52_id_front.npz",
                             kp2d_backend='dlib')  # or 'face_alignment'

# Fit id to image
src_img = cv2.imread("./test_data/matthijs_frontal.jpg")
kp2d = fs_fitter.detect_kp2d(src_img)  # extract 2D key points
mesh, params, mesh_verts_img = fs_fitter.fit_kp2d(kp2d, model)  # fit model
id, _, scale, trans, rot_vector = params

# Get texture
texture = fs_fitter.get_texture(src_img, mesh_verts_img, mesh, model)
filename = './demo_output/test_mesh.jpg'
cv2.imwrite(filename, texture)

# Save base mesh
mesh.export(output_dir='./demo_output', file_name='test_mesh', texture_name=filename, enable_vc=False, enable_vt=True)


def texcoords2vertexcolor(mesh, texture_image):
    v_count = len(mesh.vertices)
    vertex_colors = np.zeros((v_count, 3))

    idx = 0
    for tc in mesh.texcoords:

        if idx >= v_count: #b/c mesh somehow has more texcoords than vertices. # TODO look into this
            break

        # Read image at texcoord
        i = int(tc[0] * texture.shape[0])
        j = int(tc[1] * texture.shape[1])
        vertex_colors[idx, :] = texture[i, j]

        idx += 1

    return vertex_colors


mesh.vert_colors = texcoords2vertexcolor(mesh, texture)

# Read expressions from openFace recording
exp_vecs = read_open_face_expressions()
frame_count = exp_vecs.shape[1]

# Animate blendshape meshes based on OpenFace recorded expression vectors
for i in range(frame_count):
    w_exp = exp_vecs[:, i]

    mesh_full = model.gen_full(id, w_exp)
    mesh_full.vert_colors = texcoords2vertexcolor(mesh_full, texture)

    # Render image
    Rt = np.array([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 1000]], dtype=np.float64)

    tm = trimesh.Trimesh(vertices=mesh_full.vertices,
                              faces=mesh_full.faces_v - 1,
                              vertex_colors=mesh_full.vert_colors)

    depth_full, image_full = render_cvcam(tm, Rt=Rt)

    # Saving rendered image
    cv2.imwrite(f'./demo_output/{i}.png', image_full)
    print(f'Rendered frame {i}')
