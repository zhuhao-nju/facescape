import numpy as np, trimesh
from src.facescape_bm import facescape_bm
from src.renderer import render_cvcam
from src.utility import show_img_arr

np.random.seed(1000)

model = facescape_bm("./bilinear_model_v1.6/facescape_bm_v1.6_847_50_52_id_front.npz")

# create random identity vector
random_id_vec = (np.random.random(50) - 0.5) * 0.1
if random_id_vec[0] > 0:
    random_id_vec = -random_id_vec

# create random expression vector
exp_vec = np.zeros(52)
exp_vec[0] = 1
exp_vec[np.random.randint(52)] = 1

# creat random color vector
random_color_vec = (np.random.random(100) - 0.5) * 100

# generate and save full head mesh
mesh_full = model.gen_full(random_id_vec, exp_vec)
# mesh_full.export("./demo_output/bm_v16_result_full.obj")

# generate and save facial mesh
mesh_face = model.gen_face(random_id_vec, exp_vec)
# mesh_face.export("./demo_output/bm_v16_result_face.obj")

# generate and save facial mesh with rough vertex color
# mesh_face_color = model.gen_face_color(random_id_vec, exp_vec, random_color_vec)
# mesh_face_color.export("./demo_output/bm_v16_result_face_color.obj", enable_vc=True)

print("Results saved to './demo_output/'")

# render generated meshes
Rt = np.array([[1, 0, 0, 0],
               [0, -1, 0, 0],
               [0, 0, -1, 500]], dtype=np.float64)

depth_full, image_full = render_cvcam(trimesh.Trimesh(vertices=mesh_full.vertices,
                                                      faces=mesh_full.faces_v - 1),
                                      Rt=Rt)

depth_face, image_face = render_cvcam(trimesh.Trimesh(vertices=mesh_face.vertices,
                                                      faces=mesh_face.faces_v - 1),
                                      Rt=Rt)

depth_face_color, image_face_color = render_cvcam(trimesh.Trimesh(
    vertices=mesh_full.vertices,
    faces=mesh_full.faces_v - 1,
    vertex_colors=mesh_full.vert_colors),
    Rt=Rt)

# show rendered images
merge_img = np.concatenate((image_full, image_face, image_face_color), 1)

show_img_arr(merge_img, bgr_mode=True)