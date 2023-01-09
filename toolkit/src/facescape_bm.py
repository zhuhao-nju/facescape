"""
Copyright 2020, Hao Zhu, Haotian Yang, NJU.
Bilinear model.
"""

import numpy as np
from src.mesh_obj import mesh_obj

class facescape_bm(object):
    def __init__(self, filename):
        bm_model = np.load(filename, allow_pickle=True)
        self.shape_bm_core = bm_model['shape_bm_core']  # shape core

        # Calculating the residual converts the shape core into the residual representation
        sub_tensor = np.stack((self.shape_bm_core[:, 0, :],) * self.shape_bm_core.shape[1], 1)
        res_tensor = self.shape_bm_core - sub_tensor
        res_tensor[:, 0, :] = self.shape_bm_core[:, 0, :]
        self.shape_bm_core = res_tensor

        self.color_bm_core = bm_model['color_bm_core'] # color core
        self.color_bm_mean = bm_model['color_bm_mean'] # color mean

        self.fv_indices = bm_model['fv_indices'] # face - vertex indices
        self.ft_indices = bm_model['ft_indices'] # face - texture_coordinate indices
        self.fv_indices_front = bm_model['fv_indices_front'] # frontal face-vertex indices
        self.ft_indices_front = bm_model['ft_indices_front'] # frontal face-texture_coordinate indices
        
        self.vc_dict_front = bm_model['vc_dict_front'] # frontal vertex color dictionary
        self.v_indices_front = bm_model['v_indices_front'] # frontal vertex indices
        
        self.vert_num = bm_model['vert_num'] # vertex number
        self.face_num = bm_model['face_num'] # face number
        self.frontal_vert_num = bm_model['frontal_vert_num'] # frontal vertex number
        self.frontal_face_num = bm_model['frontal_face_num'] # frontal face number
        
        self.texcoords = bm_model['texcoords'] # texture coordinates (constant)
        self.facial_mask = bm_model['facial_mask'] # UV facial mask
        self.sym_dict = bm_model['sym_dict'] # symmetry dictionary
        self.lm_list_v16 = bm_model['lm_list_v16'] # landmark indices
        
        self.vert_10to16_dict = bm_model['vert_10to16_dict'] # vertex indices dictionary (v1.0 to v1.6)
        self.vert_16to10_dict = bm_model['vert_16to10_dict'] # vertex indices dictionary (v1.6 to v1.0)
        
        if 'id_mean' in bm_model.files:
            self.id_mean = bm_model['id_mean'] # identity factors mean
        if 'id_var' in bm_model.files:
            self.id_var = bm_model['id_var'] # identity factors variance
        
        # make expression GaussianMixture model
        if 'exp_gmm_weights' in bm_model.files:
            self.exp_gmm_weights = bm_model['exp_gmm_weights']
        if 'exp_gmm_means' in bm_model.files:
            self.exp_gmm_means = bm_model['exp_gmm_means']
        if 'exp_gmm_covariances' in bm_model.files:
            self.exp_gmm_covariances = bm_model['exp_gmm_covariances']
        
        if 'contour_line_right' in bm_model.files:
            self.contour_line_right = bm_model['contour_line_right'].tolist() # contour line - right
        if 'contour_line_left' in bm_model.files:
            self.contour_line_left = bm_model['contour_line_left'].tolist() # contour line - left
        if 'bottom_cand' in bm_model.files:
            self.bottom_cand = bm_model['bottom_cand'].tolist() # bottom cand

    # generate full mesh
    def gen_full(self, id_vec, exp_vec):
        verts = self.shape_bm_core.dot(id_vec).dot(exp_vec).reshape((-1, 3))
        mesh = mesh_obj()
        mesh.create(vertices = verts,
                    texcoords = self.texcoords,
                    faces_v = self.fv_indices,
                    faces_vt = self.ft_indices)
        return mesh
    
    # generate facial mesh
    def gen_face(self, id_vec, exp_vec):
        verts = self.shape_bm_core.dot(id_vec).dot(exp_vec).reshape((-1, 3))
        mesh = mesh_obj()
        mesh.create(vertices = verts[self.v_indices_front], 
                    texcoords = self.texcoords, 
                    faces_v = self.fv_indices_front, 
                    faces_vt = self.ft_indices_front)
        return mesh
    
    # generate facial mesh with vertex color
    def gen_face_color(self, id_vec, exp_vec, vc_vec):
        
        verts = self.shape_bm_core.dot(id_vec).dot(exp_vec).reshape((-1, 3))
        vert_colors = self.color_bm_mean + self.color_bm_core.dot(vc_vec)
        vert_colors = vert_colors.reshape((-1, 3)) / 255
        mesh = mesh_obj()
        
        new_vert_colors = vert_colors[self.vc_dict_front][:,[2,1,0]]
        new_vert_colors[(self.vc_dict_front == -1)] = np.array([0, 0, 0], dtype = np.float32)
        
        mesh.create(vertices = verts[self.v_indices_front], 
                    vert_colors = new_vert_colors,
                    texcoords = self.texcoords, 
                    faces_v = self.fv_indices_front, 
                    faces_vt = self.ft_indices_front)
        return mesh

