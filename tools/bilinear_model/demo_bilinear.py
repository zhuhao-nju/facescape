import numpy as np
import pickle

# triangle faces
with open('./data/predef_front_faces.pkl', 'rb') as f:
    faces_front = pickle.load(f)

with open('./data/front_indices.pkl', 'rb') as f:
    indices_front = pickle.load(f)
with open('./data/predef_faces.pkl', 'rb') as f:
    faces_full = pickle.load(f)
# texture coordinates
with open('./data/predef_texcoords.pkl', 'rb') as f:
    texcoords = pickle.load(f)

# bilinear model with 52 expression parameters and 50 identity parameters
# We perform Tucker decomposition only along the identity dimension to reserve the semantic meaning of parameters in expression dimension as speciﬁc blendshape weights
core_tensor = np.load('./data/core_847_50_52.npy')
factors_id = np.load('./data/factors_id_847_50_52.npy')

matrix_tex = np.load('./data/matrix_text_847_100.npy')
mean_tex = np.load('./data/mean_text_847_100.npy')
factors_tex = np.load('./data/factors_tex_847_100.npy')

id = factors_id[0]
exp = np.zeros(52)
exp[0] = 1

core_tensor = core_tensor.transpose((2, 1, 0))
mesh_vertices_full = core_tensor.dot(id).dot(exp).reshape((-1, 3))
mesh_vertices_front = mesh_vertices_full[indices_front]

tex = mean_tex + matrix_tex.dot(factors_tex[0])
tex = tex.reshape((-1, 3)) / 255

with open('./bilinear_result_head.obj', "w") as f:
    for i in range(mesh_vertices_full.shape[0]):
        f.write("v %.6f %.6f %.6f\n" % (mesh_vertices_full[i, 0], mesh_vertices_full[i, 1], mesh_vertices_full[i, 2]))
    for i in range(len(texcoords)):
        f.write("vt %.6f %.6f\n" % (texcoords[i][0], texcoords[i][1]))
    for face in faces_full:
        face_vertices, face_normals, face_texture_coords, material = face
        f.write("f %d/%d %d/%d %d/%d\n" % (
            face_vertices[0], face_texture_coords[0], face_vertices[1], face_texture_coords[1], face_vertices[2],
            face_texture_coords[2]))

with open('./bilinear_result_face_color.obj', "w") as f:
    for i in range(mesh_vertices_front.shape[0]):
        f.write("v %.6f %.6f %.6f %.6f %.6f %.6f\n" % (
            mesh_vertices_front[i, 0], mesh_vertices_front[i, 1], mesh_vertices_front[i, 2], tex[i, 2], tex[i, 1], tex[i, 0]))
    for face in faces_front:
        face_vertices, face_normals, face_texture_coords, material = face
        f.write("f %d %d %d\n" % (face_vertices[0], face_vertices[1], face_vertices[2]))
print("Results saved to bilinear_result_head.obj and bilinear_result_face_color.obj")
