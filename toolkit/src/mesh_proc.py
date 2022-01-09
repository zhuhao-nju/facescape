import numpy as np, pickle, json
from scipy import sparse
from scipy.linalg import orthogonal_procrustes
from scipy.sparse.linalg import lsqr

WEIGHT = 1.0

# Laplacian mesh editing
class Laplacian_deformer():
    def getLaplacianMatrixUmbrella(self, mesh, anchorsIdx):
        n = len(mesh.vertices)
        k = anchorsIdx.shape[0]
        I = []
        J = []
        V = []

        # Build sparse Laplacian Matrix coordinates and values
        for i in range(n):
            neighbors = mesh.vertex_neighbors[i]
            z = len(neighbors)
            I = I + ([i] * (z + 1))  # repeated row
            J = J + neighbors + [i]  # column indices and this row
            V = V + ([-1] * z) + [z]  # negative weights and row degree

        # augment Laplacian matrix with anchor weights
        for i in range(k):
            I = I + [n + i]
            J = J + [anchorsIdx[i]]
            V = V + [WEIGHT]  # default anchor weight

        L = sparse.coo_matrix((V, (I, J)), shape=(n + k, n)).tocsr()

        return L
    
    def solveLaplacianMesh(self, raw_mesh, anchors, anchorsIdx):
        mesh = raw_mesh.copy()
        
        vertices = np.array(mesh.vertices)
        n = vertices.shape[0]  # N x 3
        k = anchorsIdx.shape[0]

        L = self.getLaplacianMatrixUmbrella(mesh, anchorsIdx)
        delta = np.array(L.dot(vertices))

        # augment delta solution matrix with weighted anchors
        for i in range(k):
            delta[n + i, :] = WEIGHT * anchors[i, :]

        # update mesh vertices with least-squares solution
        for i in range(3):
            vertices[:, i] = lsqr(L, delta[:, i])[0]

        mesh.vertices = vertices.tolist()

        return mesh

def register_mesh(tplt_markers, markers_3d, src_mesh):
    tplt_center = np.mean(tplt_markers, axis=0)
    markers_center = np.mean(markers_3d, axis=0)
    markers_3d_centered = markers_3d - markers_center
    tplt_markers_centered = tplt_markers - tplt_center
    scale_tgt = np.linalg.norm(markers_3d_centered) / np.linalg.norm(tplt_markers_centered)
    markers_3d_centered = markers_3d_centered / scale_tgt
    translate = tplt_center
    rotation, _ = orthogonal_procrustes(tplt_markers_centered, markers_3d_centered)

    # transform verts
    src_verts = np.array(src_mesh.vertices)
    tgt_verts = src_verts.copy()
    tgt_verts = tgt_verts - markers_center
    tgt_verts = np.dot(rotation, tgt_verts.T).T
    tgt_verts = tgt_verts / scale_tgt
    tgt_verts = tgt_verts + translate

    markers_3d = markers_3d - markers_center
    markers_3d = np.dot(rotation, markers_3d.T).T
    markers_3d = markers_3d / scale_tgt
    markers_3d = markers_3d + translate

    tgt_verts_list = []
    for i in range(tgt_verts.shape[0]):
        tgt_verts_list.append([tgt_verts[i, 0], tgt_verts[i, 1], tgt_verts[i, 2]])
    src_mesh.vertices = tgt_verts_list
    return src_mesh, markers_3d


class mesh_aligner:
    def __init__(self, templates, rig_params):
        self.seam_indices = []
        self.new_vertices_idx = []
        self.new_faces = []
        self.new_seam_indices = []
        self.mesh_deformer = Laplacian_deformer()
        self.predef = templates[0].copy()
        self.seam_indices = rig_params['seam_indices']
        self.new_vertices_idx = rig_params['new_vertices_idx']
        self.new_faces = rig_params['new_faces']
        self.new_seam_indices = rig_params['new_seam_indices']
    
    def align(self, scan_neutral, scan_exp, idx):
        self.predef.vertices = np.array(scan_exp)[self.new_vertices_idx[idx]]
        self.predef.faces = self.new_faces[idx]
        movable, _ = register_mesh(np.array(scan_neutral)[self.seam_indices[idx]],
                                   np.array(scan_exp)[self.seam_indices[idx]], self.predef)
        anchors = np.array(scan_neutral)[self.seam_indices[idx]]
        deformed_movable = self.mesh_deformer.solveLaplacianMesh(movable, anchors, 
                                                                 np.array(self.new_seam_indices[idx]))
        scan_neutral = np.array(scan_neutral)
        scan_neutral[self.new_vertices_idx[idx]] = np.array(deformed_movable.vertices)
        return scan_neutral

