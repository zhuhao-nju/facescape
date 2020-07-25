import sys, pyrender, trimesh, cv2, PIL.Image, PIL.ImageOps
import numpy as np
sys.path.append("../lib/")
import renderer, utility, camera

# read model
model = trimesh.load_mesh("../../samples/sample_tu_model/1_neutral.obj", process = False)

# rotate around y-axis
model.vertices = utility.rotate_verts_y(model.vertices, 30)

# set material
model.visual.material.diffuse = np.array([255, 255, 255, 255], dtype=np.uint8)

# set K Rt
K = np.array([[1000, 0 , 249.5],
              [0, 1000, 249.5],
              [0, 0, 1]])
Rt = np.array([[1, 0 , 0, 0],
               [0, -1, 0, 0],
               [0, 0, -1, 600]])

# render img
_, rend_img = renderer.render_cvcam(K, Rt, model, scale=1.0, 
                                    std_size=(500, 500), flat_shading=True)

# read facial seg
facial_seg = PIL.Image.open("./facial_seg.png")
facial_seg_inv = PIL.ImageOps.invert(facial_seg)

# render mask
model.visual.material.image = facial_seg_inv
_, mask = renderer.render_cvcam(K, Rt, model, scale=1.0, std_size=(500, 500), flat_shading=True)

# apply mask
rend_img_m = rend_img.copy()
rend_img_m[mask>200] = 255

cv2.imwrite("./ef_result.jpg", np.concatenate((rend_img, rend_img_m), axis=1))
print("result saved to ./ef_result.jpg")
