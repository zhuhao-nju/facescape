import sys, pyrender, trimesh, cv2, openmesh
import numpy as np
sys.path.append("../lib/")
import renderer, camera

# read model 
model = trimesh.load_mesh("../../samples/sample_tu_model/1_neutral.obj", process=False)

# get vertices using openmesh, because trimesh doesn't preserve vertex number and order
om_mesh = openmesh.read_trimesh("../../samples/sample_tu_model/1_neutral.obj") 
verts = om_mesh.points()

# set material
model.visual.material.diffuse = np.array([255, 255, 255, 255], dtype=np.uint8)

# set K Rt (cv camera coordinate)
K = np.array([[2000, 0 , 499.5],
              [0, 2000, 499.5],
              [0, 0, 1]])
Rt = np.array([[1, 0 , 0, 0],
               [0, -1, 0, 0],
               [0, 0, -1, 600]])

# render
_, color = renderer.render_cvcam(K, Rt, model, scale=1.0, 
                                 std_size=(1000, 1000), flat_shading=True)

# read landmark list
lm_list = np.loadtxt("./landmark_indices.txt", dtype=np.int)


# make camera for projection
cam = camera.CamPara(K = K, Rt = Rt)

# draw landmarks
color_draw = color.copy()
for ind, lm_ind in enumerate(lm_list):
    uv = cam.project(verts[lm_ind])
    u, v = np.round(uv).astype(np.int)
    color_draw = cv2.circle(color_draw, (u, v), 10, (100, 100, 100), -1)
    color_draw = cv2.putText(color_draw, "%02d"%(ind), (u-8, v+4), 
                             fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                             fontScale = 0.4,
                             color = (255, 255, 255))

# save out
cv2.imwrite("./lm_result.jpg", color_draw)

print("results saved to './lm_result.jpg'")
