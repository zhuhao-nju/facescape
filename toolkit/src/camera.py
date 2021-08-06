"""
Copyright 2020, Hao Zhu, NJU.
Camera projection and inverse-projection.
"""

import numpy as np

# Basic camera projection and inv-projection
class CamPara():
    def __init__(self, K=None, Rt=None):
        img_size = [200,200]
        if K is None:
            K = np.array([[500, 0, 99.5],
                          [0, 500, 99.5],
                          [0, 0, 1]])
        else:
            K = np.array(K)
        if Rt is None:
            Rt = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0]])
        else:
            Rt = np.array(Rt)
        R = Rt[:,:3]
        t = Rt[:,3]
        self.cam_center = -np.dot(R.T,t)
        
        # compute projection and inv-projection matrix
        self.proj_mat = np.dot(K, Rt)
        self.inv_proj_mat = np.linalg.pinv(self.proj_mat)

        # compute ray directions of camera center pixel
        c_uv = np.array([float(K[0, 2]), float(K[1, 2])])
        self.center_dir = self.inv_project(c_uv)
            
    def get_camcenter(self):
        return self.cam_center
    
    def get_center_dir(self):
        return self.center_dir
    
    def project(self, p_xyz):
        p_xyz = np.double(p_xyz)
        p_uv_1 = np.dot(self.proj_mat, np.append(p_xyz, 1))
        if p_uv_1[2] == 0:
            return 0
        p_uv = (p_uv_1/p_uv_1[2])[:2]
        return p_uv
    
    # inverse projection, if depth is None, return a normalized direction
    def inv_project(self, p_uv, depth = None, plane_correct = False):
        p_uv = np.double(p_uv)
        p_xyz_1 = np.dot(self.inv_proj_mat, np.append(p_uv, 1))
        if p_xyz_1[3] == 0:
            return 0
        p_xyz = (p_xyz_1/p_xyz_1[3])[:3]
        p_dir = p_xyz - self.cam_center
        p_dir = p_dir / np.linalg.norm(p_dir)
        if depth is None:
            return p_dir
        else:
            if plane_correct is True:
                depth_c = depth/np.dot(self.center_dir, p_dir)
            else:
                depth_c = depth
            real_xyz = self.cam_center + p_dir * depth_c
            return real_xyz

