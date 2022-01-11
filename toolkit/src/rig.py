"""
Copyright 2021, Haotian Yang, Hao Zhu, NJU.
Expression rigging and PCA related functions.
"""

import numpy as np, trimesh, scipy, tqdm, json, os
from scipy.optimize import minimize
from scipy.sparse import coo_matrix, linalg
from src.mesh_proc import mesh_aligner

num_scan = 19
num_bs = 51

def get_M(mesh):
    result_M = np.zeros((len(mesh.faces) * 3, 3))
    for tri_index, face in enumerate(mesh.faces):
        v1 = mesh.vertices[face[0]]
        v2 = mesh.vertices[face[1]]
        v3 = mesh.vertices[face[2]]
        M = np.zeros((3, 3))
        M[:, 0] = v2 - v1
        M[:, 1] = v3 - v1
        M[:, 2] = np.cross(v2 - v1, v3 - v1)
        result_M[tri_index * 3:tri_index * 3 + 3, :] = M
    return result_M

def get_vertices(bs_M, neutral_M, tri_num, A):
    c = np.zeros((3 * tri_num, 3))
    for tri_index in range(tri_num):
        c[3 * tri_index:3 * tri_index + 3, :] = (
            np.matmul(bs_M[3 * tri_index:3 * tri_index + 3] + neutral_M[3 * tri_index:3 * tri_index + 3],
                      np.linalg.inv(neutral_M[3 * tri_index:3 * tri_index + 3]))).T
    return linalg.spsolve(A.T.dot(A), A.T.dot(c))


def optimize_bs(bs_weight, bs_M, scan_M, templates_M, tri_num, beta):
    least_sq_A = np.zeros((num_scan + num_bs, num_bs))
    least_sq_b = np.zeros((num_scan + num_bs, 3 * 3 * tri_num))
    least_sq_A[0:num_scan, :] = bs_weight
    for i in range(num_scan):
        least_sq_b[i, :] = (scan_M[i + 1] - scan_M[0]).flatten()
    
    for i in range(num_bs):
        omega = np.power(
            (1 + np.linalg.norm(templates_M[i + 1]) / 40) / (0.1 + np.linalg.norm(templates_M[i + 1]) / 40), 2)
        template_M_res = np.zeros((tri_num * 3, 3))

        for j in range(tri_num):
            template_M_res[j * 3:j * 3 + 3, :] = np.matmul(
                np.matmul(templates_M[0, j * 3:j * 3 + 3, :] + templates_M[i + 1, j * 3:j * 3 + 3, :],
                          np.linalg.inv(templates_M[0, j * 3:j * 3 + 3, :])), 
                scan_M[0, j * 3:j * 3 + 3, :]) - scan_M[0, j * 3:j * 3 + 3, :]
        least_sq_A[num_scan + i, i] = np.sqrt(omega * beta)
        least_sq_b[num_scan + i, :] = np.sqrt(omega * beta) * template_M_res.flatten()
    
    result_M = scipy.linalg.solve(np.matmul(least_sq_A.T, least_sq_A), np.matmul(least_sq_A.T, least_sq_b))
    for i in range(num_bs):
        bs_M[i] = np.reshape(result_M[i], (-1, 3))
    
def compute_res_weight(bs_weight, bs_vertices, scan_vertex, init_weight, gama):
    return np.power(np.linalg.norm(bs_vertices.dot(bs_weight) - scan_vertex), 2) + gama * np.power(
        np.linalg.norm(bs_weight - init_weight), 2)


def optimize_weight(bs_weights, bs_vertices, scan_vertices, init_weights, bounds, gama):
    for i in range(num_scan):
        init_weight = init_weights[i, :]
        bs_weight = init_weight
        scan_vertex = scan_vertices[i, :, :].flatten()
        result = minimize(compute_res_weight, bs_weight, method='L-BFGS-B', bounds=bounds,
                          args=(bs_vertices, scan_vertex, init_weight, gama), 
                          options={'ftol': 1e-5, 'maxiter': 1000})
        bs_weights[i, :] = result.x


def get_fixed_vertices(templates):
    fixed_vertices = np.zeros((num_bs, len(templates[0].vertices)), dtype=np.bool)
    for i in range(num_bs):
        bs_vertices = np.array(templates[i + 1].vertices)
        fixed_vertices[i, :] = np.sum(np.abs(bs_vertices), axis=1) == 0
    return fixed_vertices

# rig 52 expressions from 20 expressions
def rig_20to52(id_dir, tplt_dir, params_file, mesh_ver = "1.0"):
    
    with open(params_file, 'r') as f:
        rig_params = json.load(f)

    exp_list = rig_params['exp_list']
    bs_weight = np.array(rig_params['bs_weight'])
    vert_16to10_dict = rig_params["vert_16to10_dict"]
    faces_v16 = rig_params["faces_v16"]

    templates = []
    template_neutral = trimesh.load_mesh(tplt_dir + "Neutral.obj", 
                                         maintain_order=True, process = False)
    templates.append(template_neutral)

    expt = [0, 1, 4, 5]
    for i in tqdm.trange(num_bs, desc="[step 1/8] Loading templates"):
        template = trimesh.load_mesh(tplt_dir + str(i) + ".obj", 
                                     maintain_order=True, process = False)
        if i in expt:
            template.vertices = (np.array(template.vertices) - np.array(template_neutral.vertices)).tolist()
        templates.append(template)

    weight_bounds = [(0, 1)]*num_bs

    templates_M = np.zeros((num_bs + 1, len(template_neutral.faces) * 3, 3))
    for i in tqdm.trange(len(templates), desc="[step 2/8] Computing M for templates"):
        templates_M[i, :, :] = get_M(templates[i])
        if i > 0 and i - 1 not in expt:
            templates_M[i, :, :] = templates_M[i, :, :] - templates_M[0, :, :]

    print('[step 3/8] Building align_mesh')
    aligner = mesh_aligner(templates, rig_params)

    scans = []
    scan_neutral = trimesh.load_mesh(id_dir + "aligned/1_neutral.obj",  
                                     maintain_order=True, process = False)
    scans.append(scan_neutral)
    for i in tqdm.trange(num_scan, desc="[step 4/8] Loading TU models"):
        scan = trimesh.load_mesh(id_dir + "aligned/" + exp_list[i + 1] + ".obj",  
                                 maintain_order=True, process = False)
        scans.append(scan)

    if mesh_ver == "1.0":
        scans_v16 = []
        scan_neutral = trimesh.Trimesh(vertices = scan_neutral.vertices[vert_16to10_dict], 
                                       faces = faces_v16,
                                       maintain_order = True, process = False)
        for scan in scans:
            scan = trimesh.Trimesh(vertices = scan.vertices[vert_16to10_dict], 
                                   faces = faces_v16,
                                   maintain_order = True, process = False)
            scans_v16.append(scan)
        scans = scans_v16

    scan_M = np.zeros((num_scan + 1, len(scan_neutral.faces) * 3, 3))
    for i in tqdm.trange(len(scans), desc="[step 5/8] Computing M for TU models"):
        scan_M[i, :, :] = get_M(scans[i])

    print('[step 6/8] Building A')
    row = np.zeros(9 * len(scan_neutral.faces))
    column = np.zeros(9 * len(scan_neutral.faces))
    A_data = np.zeros(9 * len(scan_neutral.faces))
    for tri_index, face in enumerate(scan_neutral.faces):

        v1 = scan_neutral.vertices[face[0]]
        v2 = scan_neutral.vertices[face[1]]
        v3 = scan_neutral.vertices[face[2]]

        V = np.zeros((3, 2))
        V[:, 0] = v2 - v1
        V[:, 1] = v3 - v1
        Q, R = np.linalg.qr(V)
        affine = np.matmul(np.linalg.inv(R), Q.T)

        row[tri_index * 9:tri_index * 9 + 3] = tri_index * 3
        row[tri_index * 9 + 3:tri_index * 9 + 6] = tri_index * 3 + 1
        row[tri_index * 9 + 6:tri_index * 9 + 9] = tri_index * 3 + 2
        column[tri_index * 9:tri_index * 9 + 9:3] = face[0]
        column[tri_index * 9 + 1:tri_index * 9 + 9:3] = face[1]
        column[tri_index * 9 + 2:tri_index * 9 + 9:3] = face[2]
        A_data[tri_index * 9] = -affine[0, 0] - affine[1, 0]
        A_data[tri_index * 9 + 1: tri_index * 9 + 3] = affine[0:2, 0]
        A_data[tri_index * 9 + 3] = -affine[0, 1] - affine[1, 1]
        A_data[tri_index * 9 + 4:tri_index * 9 + 6] = affine[0:2, 1]
        A_data[tri_index * 9 + 6] = -affine[0, 2] - affine[1, 2]
        A_data[tri_index * 9 + 7:tri_index * 9 + 9] = affine[0:2, 2]

    A = coo_matrix((A_data, (row, column)), shape=(3 * len(scan_neutral.faces), len(scan_neutral.vertices))).tocsr()

    scan_neutral_vertices = np.array(scan_neutral.vertices)
    scan_vertices = np.zeros((num_scan, len(scan_neutral.vertices), 3))
    for i in range(num_scan):
        scan_vertices[i, :, :] = np.array(scans[i + 1].vertices) - scan_neutral_vertices

    bs_M = np.zeros((num_bs, len(template_neutral.faces) * 3, 3))    

    init_weights = bs_weight.copy()
    gama, beta = 20000, 0.5
    for loop in tqdm.trange(5, desc="[step 7/8] Optimizing"):
        optimize_bs(bs_weight, bs_M, scan_M, templates_M, len(scan_neutral.faces), beta)
        bs_vertices = np.zeros((3 * len(scan_neutral.vertices), num_bs))

        if loop != 4:
            for i in range(num_bs):
                vertices = get_vertices(bs_M[i], scan_M[0], len(scan_neutral.faces), A)
                vertices = aligner.align(scan_neutral_vertices, vertices, i)
                bs_vertices[:, i] = (vertices - scan_neutral_vertices).flatten()

            optimize_weight(bs_weight, bs_vertices, scan_vertices, init_weights, weight_bounds, gama)

            beta -= 0.1
            gama -= 5000

    # save results
    os.makedirs(id_dir + 'rigging', exist_ok=True)
    scan_neutral.export(id_dir + 'rigging/Neutral.obj');
    for i in tqdm.trange(num_bs, desc="[step 8/8] Saving result models"):
        vertices = get_vertices(bs_M[i], scan_M[0], len(scan_neutral.faces), A)
        vertices = aligner.align(scan_neutral_vertices, vertices, i)
        scan_neutral.vertices = vertices.tolist()
        scan_neutral.export(id_dir + 'rigging/' + str(i) + '.obj')
    
    print("Done, results saved to %srigging/" % id_dir)


# generate core of bilinear model from rigged meshes
# the shape of verts_arr should be [vert_num, id_num, exp_num]
# exp_dims=0 means PCA is not applied to the expression dimension
def make_bm(verts_arr, id_dims=50, exp_dims=0):
    import tensorly
    tensorly.set_backend('pytorch')
    
    verts_tensor = torch.from_numpy(verts_arr)
    if exp_dims == 0:
        core, factors = partial_tucker(tensor = verts_tensor, modes = [0, 1], 
                                       rank=[id_dims])
    else:
        core, factors = partial_tucker(tensor = vc_tensor, modes = [0, 1], 
                                       rank=[id_dims, exp_dims])
    
    return core, factors

