"""
Copyright 2020, Hao Zhu, Haotian Yang, NJU.
Parametric model fitter.
"""

import numpy as np, cv2, os, sklearn.mixture
from scipy.optimize import least_squares, minimize
from scipy.linalg import orthogonal_procrustes
from src.facescape_bm import facescape_bm
from src.mesh_obj import mesh_obj

class facescape_fitter(facescape_bm):
    def __init__(self, fs_file, kp2d_backend = 'dlib', 
                 dlib_kp2d_model = "./predef/shape_predictor_68_face_landmarks.dat"):
        super(facescape_fitter, self).__init__(fs_file)
        
        # make expression GaussianMixture model
        self.exp_gmm = sklearn.mixture.GaussianMixture(n_components = len(self.exp_gmm_means), 
                                                       covariance_type='full')    
        self.exp_gmm.precisions_cholesky_ = \
                            np.linalg.cholesky(np.linalg.inv(self.exp_gmm_covariances))
        self.exp_gmm.weights_ = self.exp_gmm_weights
        self.exp_gmm.means_ = self.exp_gmm_means
        self.exp_gmm.covariances_ = self.exp_gmm_covariances
        
        self.kp2d_backend = kp2d_backend
        
        # prepare landmarks extractor
        if self.kp2d_backend == 'face_alignment':
            import face_alignment
            self.detector = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, 
                                                         flip_input=False)
        elif self.kp2d_backend == 'dlib':
            import dlib
            if os.path.isfile(dlib_kp2d_model) is False:
                print("Missing model: please download the model of landmarkd predictor from " + \
                      "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2, " + \
                      "and extract 'shape_predictor_68_face_landmarks.dat' to %s" % dlib_kp2d_model)
                return 0
            self.face_pred = dlib.shape_predictor(dlib_kp2d_model)
            self.detector = dlib.get_frontal_face_detector()
        else:
            print("kp2d_backend is not recognized: '%s', " % self.kp2d_backend + \
                  "please use 'face_alignment' or 'dlib'.")
        self.fp_size = 512
        
    def detect_kp2d(self, src_img):
        fp_scale = float(np.max(src_img.shape[:2])) / self.fp_size
        sc_img = cv2.resize(src_img, (round(src_img.shape[1]/fp_scale), 
                                      round(src_img.shape[0]/fp_scale)))        
        # ========== extract landmarks ==========
        if self.kp2d_backend == 'face_alignment':
            preds = self.detector.get_landmarks(sc_img)
            if len(preds) == 0:
                print("no face found by face_alignment detector.")
                return 0
            elif len(preds) >1:
                print("more than one face found by face_alignment detector.")
                return 0
            pts = preds[0].astype(np.float64)
            kp2d = pts * fp_scale
            kp2d[:, 1] = src_img.shape[0] - kp2d[:, 1] - 1
        elif self.kp2d_backend == 'dlib':
            faces = self.detector(sc_img, 1) # detect faces
            pts = self.face_pred(sc_img, faces[0]) # get landmarks for the first face
            kp2d = np.array([[p.x*fp_scale, src_img.shape[0] - p.y*fp_scale - 1] for p in pts.parts()])
        return kp2d

    def fit_kp2d(self, kp2d):

        # ========== initialize ==========
        lm_pos = np.asarray(kp2d)
        id = self.id_mean
        exp = np.array([1] + [0] * 51, dtype=np.float32)
        rot_vector = np.array([0, 0, 0], dtype=np.double)
        trans = np.array([0, 0])
        scale = 1.

        mesh_verts = self.shape_bm_core.dot(id).dot(exp).reshape((-1, 3))
        mesh_verts_img = self.project(mesh_verts, rot_vector, scale, trans)

        lm_index = self.lm_list_v16

        # ========== iterative optimize ==========
        for optimize_loop in range(4):

            vertices_mean = np.mean(mesh_verts_img[lm_index], axis=0)
            vertices_2d = mesh_verts_img[lm_index] - vertices_mean
            lm_index_full = np.zeros(len(lm_index) * 3, dtype=int)
            for i in range(len(lm_index) * 3):
                lm_index_full[i] = lm_index[i // 3] * 3 + i % 3

            lm_mean = np.mean(lm_pos, axis=0)
            lm = lm_pos - lm_mean
            scale = np.sum(np.linalg.norm(lm, axis=1)) / np.sum(np.linalg.norm(vertices_2d, axis=1))
            trans = lm_mean - vertices_mean * scale

            lm_core_tensor = self.shape_bm_core[lm_index_full]

            lm_pos_3D = lm_core_tensor.dot(id).dot(exp).reshape((-1, 3))
            scale, trans, rot_vector = self._optimize_rigid_pos_2d(scale, trans, rot_vector,
                                                                lm_pos_3D, lm_pos)
            id = self._optimize_identity_2d(scale, trans, rot_vector, id, exp,
                                    lm_core_tensor, lm_pos, prior_weight=1)
            exp = self._optimize_expression_2d(scale, trans, rot_vector, id, exp,
                                       lm_core_tensor, lm_pos, prior_weight=1)
            mesh_verts = self.shape_bm_core.dot(id).dot(exp).reshape((-1, 3))
            mesh_verts_img = self.project(mesh_verts, rot_vector, scale, trans)

            lm_index = self._update_3d_lm_index(mesh_verts_img, lm_index)

        # ========== make mesh ==========
        mesh = mesh_obj()
        mesh.create(vertices=self.project(mesh_verts, rot_vector, scale, trans, keepz=True),
                    faces_v=self.fv_indices,   # face vertices
                    faces_vt=self.ft_indices,  # face texture coordinates
                    texcoords=self.texcoords   # uv coordinates
                    )

        params = (id, exp, scale, trans, rot_vector)

        return mesh, params, mesh_verts_img
    
    
    # input is 68 x 3 numpy array or list
    def fit_kp3d(self, lm_3d):
        
        lm_3d = np.asarray(lm_3d)
        
        lm_index_full = np.zeros(len(self.lm_list_v16) * 3, dtype=int)
        for i in range(len(self.lm_list_v16) * 3):
            lm_index_full[i] = self.lm_list_v16[i // 3] * 3 + i % 3
        lm_core_tensor = self.shape_bm_core[lm_index_full]        

        # ========== initialize ==========
        id = self.id_mean
        exp = np.array([1] + [0] * 51, dtype = np.float32) # neutral expression
        rot_vector = np.array([0, 0, 0], dtype=np.double)
        trans = np.array([0, 0])
        scale = 1.
        
        recon_lm = lm_core_tensor.dot(id).dot(exp).reshape((-1, 3))
        
        for optimize_loop in range(5):
            scale, trans, rot_matrix = self._optimize_rigid_pos_3d(recon_lm, lm_3d)
            tar_verts_align = lm_3d.copy()
            tar_verts_align = rot_matrix.T.dot((tar_verts_align - trans).T / scale).T + \
                              np.mean(recon_lm, 0)
            id = self._optimize_identity_3d(scale, trans, rot_matrix, id, exp, 
                                            lm_core_tensor, tar_verts_align)
            exp = self._optimize_expression_3d(scale, trans, rot_matrix, id, exp, 
                                               lm_core_tensor, tar_verts_align)
            recon_lm = lm_core_tensor.dot(id).dot(exp).reshape((-1, 3))
        
        recon_verts = self.shape_bm_core.dot(id).dot(exp).reshape((-1, 3))
        scale, trans, rot_matrix = self._optimize_rigid_pos_3d(recon_lm, lm_3d)
        recon_verts = rot_matrix.dot((recon_verts - np.mean(recon_lm, 0)).T * scale).T + trans
        
        # ========== make mesh ==========
        mesh = mesh_obj()
        mesh.create(vertices = recon_verts, 
                    faces_v = self.fv_indices_front)
        
        params = (id, exp, scale, trans, rot_vector)

        return mesh, params

    def get_texture(self, img, verts_img, mesh):
        """ Generates texture map from face image. This code is adapted to work with the v1.6 model
        source: https://github.com/yanght321/Detailed3DFace/

        :param img: source image
        :param verts_img: as returned by fit_kp2d
        :param mesh: as returned by fit_kp2d
        :return: unwrapped texture image
        """

        h, w, _ = img.shape

        texture = np.zeros((4096, 4096, 3))

        for face in mesh.faces:
            face_vertices, face_normals, tc, material = face

            if max(abs(self.texcoords[tc[0] - 1][0] - self.texcoords[tc[1] - 1][0]),
                    abs(self.texcoords[tc[0] - 1][0] - self.texcoords[tc[2] - 1][0]),
                    abs(self.texcoords[tc[1] - 1][0] - self.texcoords[tc[2] - 1][0]),
                    abs(self.texcoords[tc[0] - 1][1] - self.texcoords[tc[1] - 1][1]),
                    abs(self.texcoords[tc[0] - 1][1] - self.texcoords[tc[2] - 1][1]),
                    abs(self.texcoords[tc[1] - 1][1] - self.texcoords[tc[2] - 1][1])) > 0.3:
                continue

            tri1 = np.float32([[[(h - int(verts_img[face_vertices[0] - 1, 1])),
                                    int(verts_img[face_vertices[0] - 1, 0])],
                                [(h - int(verts_img[face_vertices[1] - 1, 1])),
                                    int(verts_img[face_vertices[1] - 1, 0])],
                                [(h - int(verts_img[face_vertices[2] - 1, 1])),
                                    int(verts_img[face_vertices[2] - 1, 0])]]])
            tri2 = np.float32(
                [[[4096 - self.texcoords[tc[0] - 1][1] * 4096, self.texcoords[tc[0] - 1][0] * 4096],
                    [4096 - self.texcoords[tc[1] - 1][1] * 4096, self.texcoords[tc[1] - 1][0] * 4096],
                    [4096 - self.texcoords[tc[2] - 1][1] * 4096, self.texcoords[tc[2] - 1][0] * 4096]]])
            r1 = cv2.boundingRect(tri1)
            r2 = cv2.boundingRect(tri2)

            tri1Cropped = []
            tri2Cropped = []

            for i in range(0, 3):
                tri1Cropped.append(((tri1[0][i][1] - r1[1]), (tri1[0][i][0] - r1[0])))
                tri2Cropped.append(((tri2[0][i][1] - r2[1]), (tri2[0][i][0] - r2[0])))

            # Apply warpImage to small rectangular patches
            img1Cropped = img[r1[0]:r1[0] + r1[2], r1[1]:r1[1] + r1[3]]
            warpMat = cv2.getAffineTransform(np.float32(tri1Cropped), np.float32(tri2Cropped))

            # Get mask by filling triangle
            mask = np.zeros((r2[2], r2[3], 3), dtype=np.float32)
            cv2.fillConvexPoly(mask, np.int32(tri2Cropped), (1.0, 1.0, 1.0), 16, 0)

            # Apply the Affine Transform just found to the src image
            img2Cropped = cv2.warpAffine(img1Cropped, warpMat, (r2[3], r2[2]), None, flags=cv2.INTER_LINEAR,
                                            borderMode=cv2.BORDER_REFLECT_101)

            # Apply mask to cropped region
            img2Cropped = img2Cropped * mask

            # Copy triangular region of the rectangular patch to the output image
            texture[r2[0]:r2[0] + r2[2], r2[1]:r2[1] + r2[3]] = texture[r2[0]:r2[0] + r2[2],
                                                                r2[1]:r2[1] + r2[3]] * ((1.0, 1.0, 1.0) - mask)

            texture[r2[0]:r2[0] + r2[2], r2[1]:r2[1] + r2[3]] = texture[r2[0]:r2[0] + r2[2],
                                                                r2[1]:r2[1] + r2[3]] + img2Cropped

        return texture


        
    # ================================= inner functions ==================================
    def _rotate(self, points, rot_vec):
        """Rotate points by given rotation vectors.
        Rodrigues' rotation formula is used.
        """
        theta = np.linalg.norm(rot_vec)
        with np.errstate(invalid='ignore'):
            v = rot_vec / theta
            v = np.nan_to_num(v)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        return cos_theta * points + sin_theta * np.cross(v, points) + \
               (points.dot(v.T) * (1 - cos_theta)).dot(v)

    def project(self, points, rot_vec, scale, trans, keepz=False):
        points_proj = self._rotate(points, rot_vec.reshape(1, 3))
        points_proj = points_proj * scale
        if keepz:
            points_proj[:, 0:2] = points_proj[:, 0:2] + trans
        else:
            points_proj = points_proj[:, 0:2] + trans
        return points_proj

    def _compute_res_rigid(self, params, lm_pos_3D, lm_pos):
        lm_pos_3D = lm_pos_3D.reshape(-1, 3)
        lm_proj = self.project(lm_pos_3D, params[3:6], params[0], params[1:3])
        return lm_proj.ravel() - lm_pos

    def _optimize_rigid_pos_2d(self, scale, trans, rot_vector, lm_pos_3D, lm_pos):
        lm_pos_3D = lm_pos_3D.ravel()
        lm_pos = lm_pos.ravel()
        params = np.hstack((scale, trans, rot_vector))
        result = least_squares(self._compute_res_rigid, params, verbose=0, 
                               x_scale='jac', ftol=1e-5, method='lm',
                               args=(lm_pos_3D, lm_pos))
        return result.x[0], result.x[1:3], result.x[3:6]

    def _optimize_rigid_pos_3d(self, recon_verts, tar_verts):
        tar_center = np.mean(tar_verts, axis=0)
        recon_center = np.mean(recon_verts, axis=0)
        tar_verts_centered = tar_verts - tar_center
        recon_verts_centered = recon_verts - recon_center
        scale_recon = np.linalg.norm(recon_verts_centered) / np.linalg.norm(tar_verts_centered)
        recon_verts_centered = recon_verts / scale_recon
        translate = tar_center
        rotation, _ = orthogonal_procrustes(tar_verts_centered, recon_verts_centered)
        return 1 / scale_recon, translate, rotation    

    def _compute_res_id_2d(self, id, id_matrix, scale, trans, rot_vector, lm_pos, prior_weight):
        id_matrix = id_matrix.reshape(-1, id.shape[0])
        lm_pos_3D = id_matrix.dot(id).reshape((-1, 3))
        lm_proj = self.project(lm_pos_3D, rot_vector, scale, trans).ravel()
        return np.linalg.norm(lm_proj - lm_pos) ** 2 / scale / scale + \
               prior_weight * (id - self.id_mean).dot(np.diag(1 / \
               self.id_var)).dot(np.transpose([id - self.id_mean]))
    
    def _optimize_identity_2d(self, scale, trans, rot_vector, id, exp, 
                           lm_core_tensor, lm_pos, prior_weight=20):
        id_matrix = np.tensordot(lm_core_tensor, exp, axes=([1], [0])).ravel()
        lm_pos = lm_pos.ravel()
        result = minimize(self._compute_res_id_2d, id, method='L-BFGS-B',
                          args=(id_matrix, scale, trans, 
                                rot_vector, lm_pos, prior_weight), options={'maxiter': 100})
        return result.x

    def _compute_res_id_3d(self, id, id_matrix, scale, trans, rot_matrix, tar_verts):
        id_matrix = id_matrix.reshape(-1, id.shape[0])
        recon_verts = id_matrix.dot(id).reshape((-1, 3))
        recon_verts = recon_verts.ravel()
        return np.linalg.norm(recon_verts - tar_verts) ** 2 + 20 * \
               (id - self.id_mean).dot(np.diag(1 / self.id_var)).dot(np.transpose([id - self.id_mean]))

    def _optimize_identity_3d(self, scale, trans, rot_matrix, id, exp, core_tensor, tar_verts):
        id_matrix = np.tensordot(core_tensor, exp, axes=([1], [0])).ravel()
        tar_verts = tar_verts.ravel()
        result = minimize(self._compute_res_id_3d, id, method='L-BFGS-B', 
                          args=(id_matrix, scale, trans, rot_matrix, tar_verts),
                          options={'maxiter': 100})
        return result.x
    
    def _compute_res_exp_2d(self, exp, exp_matrix, scale, trans, rot_vector, lm_pos, prior_weight):
        exp_matrix = exp_matrix.reshape(-1, exp.shape[0] + 1)
        exp_full = np.ones(52)
        exp_full[1:52] = exp
        lm_pos_3D = exp_matrix.dot(exp_full).reshape((-1, 3))
        lm_proj = self.project(lm_pos_3D, rot_vector, scale, trans).ravel()

        return np.linalg.norm(lm_proj - lm_pos) ** 2 / scale / scale - prior_weight * \
               self.exp_gmm.score_samples(exp.reshape(1, -1))[0]
    
    def _optimize_expression_2d(self, scale, trans, rot_vector, id, exp, 
                             lm_core_tensor, lm_pos, prior_weight=0.02):
        exp_matrix = np.dot(lm_core_tensor, id).ravel()
        lm_pos = lm_pos.ravel()
        bounds = []
        for i in range(exp.shape[0] - 1):
            bounds.append((0, 1))
        result = minimize(self._compute_res_exp_2d, exp[1:52], method='L-BFGS-B', bounds=bounds,
                          args=(exp_matrix, scale, trans, 
                                rot_vector, lm_pos, prior_weight), options={'maxiter': 100})
        exp_full = np.ones(52)
        exp_full[1:52] = result.x
        return exp_full

    def _compute_res_exp_3d(self, exp, exp_matrix, scale, trans, rot_matrix, tar_verts):
        exp_matrix = exp_matrix.reshape(-1, exp.shape[0] + 1)
        exp_full = np.ones(52)
        exp_full[1:52] = exp
        recon_verts = exp_matrix.dot(exp_full).reshape((-1, 3))
        recon_verts = recon_verts.ravel()
        return np.linalg.norm(recon_verts - tar_verts) ** 2

    def _optimize_expression_3d(self, scale, trans, rot_matrix, id, exp, core_tensor, tar_verts):
        exp_matrix = np.dot(core_tensor, id).ravel()
        tar_verts = tar_verts.ravel()
        bounds = []
        for i in range(exp.shape[0] - 1):
            bounds.append((0, 1))
        result = minimize(self._compute_res_exp_3d, exp[1:52], method='L-BFGS-B', bounds=bounds,
                          args=(exp_matrix, scale, trans, rot_matrix, tar_verts), 
                          options={'maxiter': 100})
        exp_full = np.ones(52)
        exp_full[1:52] = result.x
        return exp_full

    def _update_3d_lm_index(self, points_proj, lm_index):
        updated_lm_index = list(lm_index)
        modify_key_right = range(9, 17)
        modify_key_left = range(0, 8)

        # get the outest point on the contour line
        for i in range(len(modify_key_right)):
            if len(self.contour_line_right[i]) != 0:
                max_ind = np.argmax(points_proj[self.contour_line_right[i], 0])
                updated_lm_index[modify_key_right[i]] = self.contour_line_right[i][max_ind]

        for i in range(len(modify_key_left)):
            if len(self.contour_line_left[i]) != 0:
                min_ind = np.argmin(points_proj[self.contour_line_left[i], 0])
                updated_lm_index[modify_key_left[i]] = self.contour_line_left[i][min_ind]

        updated_lm_index[8] = self.bottom_cand[np.argmin((points_proj[self.bottom_cand, 1]))]

        return updated_lm_index
