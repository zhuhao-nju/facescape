## ================================================================================
## Description: read predicted results (model and optional parameters) of different
##              methods, and transform them into camera coordinate. 
##
## Arguments: dir  - string, folder of the stored results.
##            fn   - string, filename.
##            f_gt - float, focal length. If f_gt=-1, use orthogonal projection; 
##                  if f>0, use perspective projection with focal length = f.
##
## Authors: Hao Zhu (zhuhaoese@nju.edu.cn), Longwei Guo
##
## License: MIT
## ================================================================================

import trimesh, numpy as np, os
from mesh_util import load_ori_mesh
import json

gt_img_size = 256
render_bias = 2.

R_pers2ortho = np.array([[1, 0, 0], 
                         [0, -1, 0], 
                         [0, 0, -1]], dtype = np.float64)

R_gl2ortho = np.array([[-1, 0, 0], 
                       [0, 1, 0], 
                       [0, 0, 1]], dtype = np.float64)

# zhuhao
def align_facescape_opti(dir, fn, f_gt=-1):
    
    try:
        src_mesh = load_ori_mesh(dir + "%s.obj" % fn)
        
        pred_align_mesh = src_mesh.copy()
        
        # move to orthogonal camera coordinate
        pred_align_mesh.vertices[:, :2] -= (gt_img_size/2)
        pred_align_mesh.vertices /= (gt_img_size/2)
        
        if f_gt > 0:
            # orthogonal to perspective
            pred_align_mesh.vertices = R_pers2ortho.dot(pred_align_mesh.vertices.T).T
            pred_align_mesh.vertices[:, 2] -= np.mean(pred_align_mesh.vertices[:,2])
            pred_align_mesh.vertices[:, 2] += (f_gt / (gt_img_size/2))
            
            # scale to make face width = 1
            pred_align_mesh.vertices /= (np.max(pred_align_mesh.vertices[:, 1]) - \
                                         np.min(pred_align_mesh.vertices[:, 1]))/3
            
        else:
            # keep orthogonal camera, add bias to make pyrender work
            pred_align_mesh.vertices[:,2] -= (np.mean(pred_align_mesh.vertices[:,2]) + render_bias)
        
        return pred_align_mesh
    except:
        pred_align_mesh = None
    return pred_align_mesh
    
# zhuhao
def align_facescape_deep(dir, fn, f_gt=-1):
    
    # read mesh
    src_mesh = load_ori_mesh(dir + "%s.obj" % fn)
    tform = np.load(dir + "%s_tform.npy" % fn, allow_pickle = True)
    
    pred_align_mesh = src_mesh.copy()
    
    # align to warp_img
    pred_align_mesh.vertices /= (gt_img_size/2)
    pred_align_mesh.vertices[:,:] -= 1
    
    # align to original image
    M_inv = np.linalg.inv(tform.item().params)
    scale = np.linalg.norm([M_inv[0, 0], M_inv[0, 1]])
    
    # move to image coordinate
    pred_align_mesh.vertices = R_pers2ortho.dot(pred_align_mesh.vertices.T).T
    pred_align_mesh.vertices[:, :2] += 1
    
    # what cv2.warpAffine() does
    pred_align_mesh.vertices[:, :2] = np.dot(M_inv[:2, :2], pred_align_mesh.vertices[:, :2].T).T + \
                                      (M_inv[:2, 2] / (gt_img_size/2))
    
    # move back to orthogonal camera coordinate
    pred_align_mesh.vertices[:, :2] -= 1
    pred_align_mesh.vertices = R_pers2ortho.T.dot(pred_align_mesh.vertices.T).T
    
    # compensate Z
    pred_align_mesh.vertices[:, 2] *= scale
    
    if f_gt > 0:
        # orthogonal to perspective
        pred_align_mesh.vertices = R_pers2ortho.dot(pred_align_mesh.vertices.T).T
        pred_align_mesh.vertices[:, 2] -= np.mean(pred_align_mesh.vertices[:,2])
        pred_align_mesh.vertices[:, 2] += (f_gt / (gt_img_size/2))
        
        # scale to make face width = 1
        pred_align_mesh.vertices /= (np.max(pred_align_mesh.vertices[:, 1]) - \
                                     np.min(pred_align_mesh.vertices[:, 1]))/3
    else:
        # keep orthogonal camera, add bias to make pyrender work
        pred_align_mesh.vertices[:,2] -= (np.mean(pred_align_mesh.vertices[:,2]) + render_bias)
    
    return pred_align_mesh


# zhuhao
def align_3ddfav2(dir, fn, f_gt=-1):
    
    # read mesh
    pred_world_mesh_path = os.path.join(dir, "%s.obj" % fn)
    pred_world_mesh = load_ori_mesh(pred_world_mesh_path)
    pred_align_mesh = pred_world_mesh.copy()
    
    # move to orthogonal camera coordinate
    pred_align_mesh.vertices /= (gt_img_size/2)
    pred_align_mesh.vertices[:, :2] -= 1
    
    if f_gt > 0:
        # orthogonal to perspective
        pred_align_mesh.vertices = R_pers2ortho.dot(pred_align_mesh.vertices.T).T
        pred_align_mesh.vertices[:, 2] -= np.mean(pred_align_mesh.vertices[:,2])        
        pred_align_mesh.vertices[:, 2] += (f_gt / (gt_img_size/2))
                
        # scale to make face width = 1
        pred_align_mesh.vertices /= (np.max(pred_align_mesh.vertices[:, 1]) - \
                                     np.min(pred_align_mesh.vertices[:, 1]))/3
    else:
        # keep orthogonal camera, add bias to make pyrender work
        pred_align_mesh.vertices[:,2] -= (np.mean(pred_align_mesh.vertices[:,2]) + render_bias)
    return pred_align_mesh


# zhuhao
def align_deca(dir, fn, f_gt=-1):
    
    # read mesh
    src_mesh = load_ori_mesh(dir + "%s_detail.obj" % fn)
    tform = np.load(dir + "%s_tform.npy" % fn, allow_pickle = True)
    cam = np.load(dir + "%s_cam.npy" % fn, allow_pickle = True)
    
    pred_align_mesh = src_mesh.copy()

    # align to warp_img
    pred_align_mesh.vertices[:, :2] += cam[0, 1:]
    pred_align_mesh.vertices *= cam[0, 0]
    
    # align to original image
    M_inv = np.linalg.inv(tform.item().params)
    scale = np.linalg.norm([M_inv[0, 0], M_inv[0, 1]])

    # move to image coordinate
    pred_align_mesh.vertices = R_pers2ortho.dot(pred_align_mesh.vertices.T).T
    pred_align_mesh.vertices[:, :2] += 1

    # what cv2.warpAffine() does
    pred_align_mesh.vertices[:, :2] = np.dot(M_inv[:2, :2], pred_align_mesh.vertices[:, :2].T).T \
                                      + (M_inv[:2, 2] / 112)
    
    # move back to orthogonal camera coordinate
    pred_align_mesh.vertices[:, :2] -= (gt_img_size / 224)
    pred_align_mesh.vertices = R_pers2ortho.T.dot(pred_align_mesh.vertices.T).T
    pred_align_mesh.vertices[:, :] *= (224 / gt_img_size)
    
    # compensate Z
    pred_align_mesh.vertices[:, 2] *= scale  
    
    if f_gt > 0:
        # orthogonal to perspective
        pred_align_mesh.vertices = R_pers2ortho.dot(pred_align_mesh.vertices.T).T
        pred_align_mesh.vertices[:, 2] -= np.mean(pred_align_mesh.vertices[:,2])
        pred_align_mesh.vertices[:, 2] += (f_gt / (gt_img_size/2))
        
        # scale to make face width = 1
        pred_align_mesh.vertices /= (np.max(pred_align_mesh.vertices[:, 1]) - \
                                     np.min(pred_align_mesh.vertices[:, 1]))/3
    else:
        # keep orthogonal camera, add bias to make pyrender work
        pred_align_mesh.vertices[:,2] -= (np.mean(pred_align_mesh.vertices[:,2]) + render_bias)
    
    return pred_align_mesh


# longwei (zhuhao checked)
def align_ext3df(dir, fn, f_gt=-1):

    try:
        # load mesh and crop information
        pred_world_mesh_path = os.path.join(dir, "%s_withBump_aligned.ply" % fn)
        pred_world_mesh = load_ori_mesh(pred_world_mesh_path)
        pred_align_mesh = pred_world_mesh.copy()

        with open(os.path.join(dir, "extreme3d_dict.json"), 'r') as f:
            coors_dict = json.load(f)
        bl = coors_dict[fn][0]
        bt = coors_dict[fn][1]
        nw = coors_dict[fn][2]
        nh = coors_dict[fn][3]

        # compensate cx cy and bl bt
        pred_align_mesh.vertices *= (nw + nh)/2./500.
        comp_cx = (nw + nh)/4 - 127.5
        comp_cy = (nw + nh)/4 - 127.5
        pred_align_mesh.vertices[:,0] += (bl + comp_cx)
        pred_align_mesh.vertices[:,1] -= (bt + comp_cy)
        
        if f_gt > 0:
            # glcam to cvcam
            pred_align_mesh.vertices = np.dot(R_pers2ortho, pred_align_mesh.vertices.T).T
            
            # scale to fit focal length
            mean_z = np.mean(pred_align_mesh.vertices[:,2])
            pred_align_mesh.vertices[:,2] -= mean_z
            pred_align_mesh.vertices *= (500/f_gt)
            pred_align_mesh.vertices[:,2] += mean_z
            
            # scale to make face width = 1
            pred_align_mesh.vertices /= (np.max(pred_align_mesh.vertices[:, 1]) - \
                                         np.min(pred_align_mesh.vertices[:, 1]))/3
            
        else:
            # perspective to orthogonal
            pred_align_mesh.vertices *= (500 / abs(np.mean(pred_align_mesh.vertices[:,2])))
            
            # move to orthogonal camera coordinate
            pred_align_mesh.vertices = pred_align_mesh.vertices/(gt_img_size/2)

            # keep orthogonal camera, add bias to make pyrender work
            pred_align_mesh.vertices[:,2] -= (np.mean(pred_align_mesh.vertices[:,2]) + render_bias)
    except:
        pred_align_mesh = None
    return pred_align_mesh

# zhuhao
def align_mgcnet(dir, fn, f_gt=-1):
    
    # read mesh
    src_mesh = load_ori_mesh(dir + "%s.ply" % fn)
    
    # read affine matrix
    M_inv = np.load(dir + "%s_Minv.npy" % fn)
    scale = np.sqrt(pow(M_inv[0, 0], 2) + pow(M_inv[1, 0], 2))
    
    # read intrinsic and extrinsic parameters
    with open(dir + "%s_cam.txt" % fn, 'r') as f:
        params = f.readlines()
    ext_vec = np.array(params[4].split(','), dtype = np.float64)
    rot_vec = ext_vec[:3]
    trans_vec = ext_vec[3:]
    int_mat = np.array(params[2].split(','), dtype = np.float64).reshape((3, 3))
    f = int_mat[0][0]
    
    # ===== align to warpped image =====
    rot_mat = rotMtx_eular(np.flip(rot_vec))
    this_Rt = np.concatenate((np.eye(3), np.zeros((3, 1))), axis = 1)
    tgt_mesh = src_mesh.copy()
    tgt_mesh.vertices = np.dot(rot_mat, tgt_mesh.vertices.T).T + trans_vec
    
    # ===== align to original image =====
    pred_align_mesh = tgt_mesh.copy()

    # move to image coordinate
    pred_align_mesh.vertices[:,:2] += (114 / f * np.mean(pred_align_mesh.vertices[:, 2]))

    # affine warp
    pred_align_mesh.vertices[:, :2] = np.dot(M_inv[:2, :2], pred_align_mesh.vertices[:, :2].T).T
    pred_align_mesh.vertices[:, :2] += (M_inv[:2, 2] / f * np.mean(pred_align_mesh.vertices[:, 2]))

    # move back to camera coordinate
    pred_align_mesh.vertices[:, :2] -= (114 / f * np.mean(pred_align_mesh.vertices[:, 2]))
    pred_align_mesh.vertices[:, :2] /= scale

    # inverse scale by adding delta Z
    pred_align_mesh.vertices[:, 2] -= np.mean(pred_align_mesh.vertices[:, 2])*(1 - 1/scale)
    
    # compensate cx cy
    pred_align_mesh.vertices[:, :2] -= (((gt_img_size/2)-112)/f*np.mean(pred_align_mesh.vertices[:, 2]))
    
    if f_gt > 0:
        # scale to fit focal length
        mean_z = np.mean(pred_align_mesh.vertices[:,2])
        pred_align_mesh.vertices[:,2] -= mean_z
        pred_align_mesh.vertices *= (f/f_gt)
        pred_align_mesh.vertices[:,2] += mean_z
        
        # scale to make face width = 1
        pred_align_mesh.vertices /= (np.max(pred_align_mesh.vertices[:, 1]) - \
                                     np.min(pred_align_mesh.vertices[:, 1]))/3
    else:
        # perspective to orthogonal
        pred_align_mesh.vertices *= (f * 2 / gt_img_size) / np.mean(pred_align_mesh.vertices[:,2])
        pred_align_mesh.vertices = R_pers2ortho.dot(pred_align_mesh.vertices.T).T
        
        # add bias to make pyrender work
        pred_align_mesh.vertices[:,2] -= (np.mean(pred_align_mesh.vertices[:,2]) + render_bias)
        
    return pred_align_mesh

# longwei
def align_prnet(dir, fn, f_gt=-1):
    
    # read mesh
    pred_world_mesh_path = os.path.join(dir, "%s.obj" % fn)
    pred_world_mesh = load_ori_mesh(pred_world_mesh_path)
    pred_align_mesh = pred_world_mesh.copy()
    
    # move to orthogonal camera coordinate
    pred_align_mesh.vertices /= (gt_img_size/2)
    pred_align_mesh.vertices[:, :2] -= 1
    
    if f_gt > 0:
        # orthogonal to perspective
        pred_align_mesh.vertices = R_pers2ortho.dot(pred_align_mesh.vertices.T).T
        pred_align_mesh.vertices[:, 2] -= np.mean(pred_align_mesh.vertices[:,2])
        pred_align_mesh.vertices[:, 2] += (f_gt / (gt_img_size/2))
        
        # scale to make face width = 1
        pred_align_mesh.vertices /= (np.max(pred_align_mesh.vertices[:, 1]) - \
                                     np.min(pred_align_mesh.vertices[:, 1]))/3
    else:
        # keep orthogonal camera, add bias to make pyrender work
        pred_align_mesh.vertices[:,2] -= (np.mean(pred_align_mesh.vertices[:,2]) + render_bias)
    
    return pred_align_mesh

# zhuhao
def align_ringnet(dir, fn, f_gt=-1):
    
    src_mesh = load_ori_mesh(dir + "%s.obj" % fn)
    params = np.load(dir + "%s.npy" % fn, allow_pickle=True, encoding='latin1')
    f_norm, cx, cy = params.item()['cam']

    # move to camera coordinate
    pred_align_mesh = src_mesh.copy()
    pred_align_mesh.vertices += np.array([cx, cy, gt_img_size / 224.])
    
    if f_gt > 0:
        # scale to fit focal length
        mean_z = np.mean(pred_align_mesh.vertices[:,2])
        pred_align_mesh.vertices[:,2] -= mean_z
        pred_align_mesh.vertices *= (f_norm*128/f_gt)
        pred_align_mesh.vertices[:,2] += mean_z
        
        # scale to make face width = 1
        pred_align_mesh.vertices /= (np.max(pred_align_mesh.vertices[:, 1]) - \
                                     np.min(pred_align_mesh.vertices[:, 1]))/3    
    else:
        # perspective to orthogonal
        pred_align_mesh.vertices *= (f_norm / np.mean(pred_align_mesh.vertices[:,2]))
        pred_align_mesh.vertices = R_pers2ortho.dot(pred_align_mesh.vertices.T).T
        
        # add bias to make pyrender work
        pred_align_mesh.vertices[:,2] -= (np.mean(pred_align_mesh.vertices[:,2]) + render_bias)
    return pred_align_mesh

# zhuhao
def align_sadrnet(dir, fn, f_gt=-1):
    
    # read mesh
    src_mesh = load_ori_mesh(os.path.join(dir, "%s.obj" % fn))
    pred_align_mesh = src_mesh.copy()
    
    # move to orthogonal camera coordinate
    pred_align_mesh.vertices[:, :2] -= 1
    pred_align_mesh.vertices[:, 1] *= -1
    pred_align_mesh.faces = pred_align_mesh.faces[:, [1, 0, 2]] # invert faces
    
    if f_gt > 0:
        # orthogonal to perspective
        pred_align_mesh.vertices = R_pers2ortho.dot(pred_align_mesh.vertices.T).T
        pred_align_mesh.vertices[:, 2] -= np.mean(pred_align_mesh.vertices[:,2])
        pred_align_mesh.vertices[:, 2] += (f_gt / (gt_img_size/2))
        
        # scale to make face width = 1
        pred_align_mesh.vertices /= (np.max(pred_align_mesh.vertices[:, 1]) - \
                                     np.min(pred_align_mesh.vertices[:, 1]))/3   
    else:
        # keep orthogonal camera, add bias to make pyrender work
        pred_align_mesh.vertices[:,2] -= (np.mean(pred_align_mesh.vertices[:,2]) + render_bias)
    
    return pred_align_mesh    

# zhuhao
def align_udl(dir, fn, f_gt=-1):
    try:
        src_mesh = load_ori_mesh(dir + "%s_fine.ply" % fn)
        param_pose = np.load(dir + "%s_pose.npy" % fn, 
                             allow_pickle = True, encoding = 'latin1')
        param_crop = np.load(dir + "%s_crop.npy" % fn, 
                             allow_pickle = True, encoding = 'latin1')

        pred_align_mesh = src_mesh.copy()

        # apply Rt
        R, t = calc_trans_matrix(param_pose[0])
        pred_align_mesh.vertices = np.dot(R, pred_align_mesh.vertices.T).T + t

        # apply transform of crop
        st_x, en_x, st_y, en_y, pad_t, pad_l = param_crop
        delta_cx = (st_x + en_x) / 2 - (gt_img_size/2)
        delta_cy = (st_y + en_y) / 2 - (gt_img_size/2)
        scale_crop = (en_x - st_x) / 300

        pred_align_mesh.vertices[:, :] *= scale_crop
        pred_align_mesh.vertices[:, :2] += (np.array([delta_cx, -delta_cy]))
        
        # align to orthogonal scale
        pred_align_mesh.vertices /= (gt_img_size/2)
    
        if f_gt > 0:
            # orthogonal to perspective
            pred_align_mesh.vertices = R_pers2ortho.dot(pred_align_mesh.vertices.T).T
            pred_align_mesh.vertices[:, 2] -= np.mean(pred_align_mesh.vertices[:,2])
            pred_align_mesh.vertices[:, 2] += (f_gt / (gt_img_size/2))

            # scale to make face width = 1
            pred_align_mesh.vertices /= (np.max(pred_align_mesh.vertices[:, 1]) - \
                                         np.min(pred_align_mesh.vertices[:, 1]))/3   
        else:
            # keep orthogonal camera, add bias to make pyrender work
            pred_align_mesh.vertices[:,2] -= (np.mean(pred_align_mesh.vertices[:,2]) + render_bias)
    
    except:
        pred_align_mesh = None
    return pred_align_mesh

# longwei
def align_df2net(dir, fn, f_gt=-1):
    # The mesh DF2Net predicted is aligned to the cropped images.
    # DF2Net_coors.json contains the source image shape, the width and height of cropped pixels.
    
    R_img2ortho = np.array([[0, 1, 0], 
                            [-1, 0, 0],
                            [0, 0, 1]])
    try:
        # load mesh and crop information
        pred_world_mesh_path = os.path.join(dir, "%s.obj" % fn)
        pred_world_mesh = load_ori_mesh(pred_world_mesh_path)
        pred_align_mesh = pred_world_mesh.copy()
        
        with open(os.path.join(dir, "DF2Net_coors.json"), 'r') as f:
            coors_dict = json.load(f)
        shape = coors_dict[fn][0]
        x_r = coors_dict[fn][1]
        y_r = coors_dict[fn][2]
        
        # move to orthogonal coordinate
        pred_align_mesh.vertices /= 512
        pred_align_mesh.vertices = R_img2ortho.dot(pred_align_mesh.vertices.T).T
        pred_align_mesh.vertices[:,1] += 1
        
        # apply transform of crop
        pred_align_mesh.vertices[:,0] = pred_align_mesh.vertices[:,0] * shape[1] + x_r
        pred_align_mesh.vertices[:,1] = pred_align_mesh.vertices[:,1] * shape[0]
        pred_align_mesh.vertices[:,2] = pred_align_mesh.vertices[:,2] * (shape[0] + shape[1])/2
        
        # align to orthogonal coordinate
        pred_align_mesh.vertices /= (gt_img_size/2)
        pred_align_mesh.vertices[:, :2] -= 1
        
        if f_gt > 0:
            # orthogonal to perspective
            pred_align_mesh.vertices = R_pers2ortho.dot(pred_align_mesh.vertices.T).T
            pred_align_mesh.vertices[:, 2] -= np.mean(pred_align_mesh.vertices[:,2])
            pred_align_mesh.vertices[:, 2] += (f_gt / (gt_img_size/2))

            # scale to make face width = 1
            pred_align_mesh.vertices /= (np.max(pred_align_mesh.vertices[:, 1]) - \
                                         np.min(pred_align_mesh.vertices[:, 1]))/3   
        else:
            # keep orthogonal camera, add bias to make pyrender work
            pred_align_mesh.vertices[:,2] -= (np.mean(pred_align_mesh.vertices[:,2]) + render_bias)
        
        pred_align_mesh.faces = pred_align_mesh.faces[:, [1, 0, 2]] # invert faces
        
    except:
        pred_align_mesh = None
    return pred_align_mesh

# longwei (checked by zhuhao)
def align_deep3df(dir, fn, f_gt=-1):
    try:
        # load mesh and crop information
        with open(os.path.join(dir, "Deep3D_cons_dict.json"), 'r') as f:
            coors_dict = json.load(f)
        transform_params = coors_dict[fn]

        w0 = np.array(transform_params[0])
        h0 = np.array(transform_params[1])
        scale = np.array(transform_params[2])
        t0 = np.array(transform_params[3])
        t1 = np.array(transform_params[4])

        w = (w0 * scale).astype(np.int32)
        h = (h0 * scale).astype(np.int32)
        target_size = 224
        left = (w/2 - target_size/2 + float((t0 - w0/2)*scale)).astype(np.int32)
        up = (h/2 - target_size/2 + float((h0/2 - t1)*scale)).astype(np.int32)

        pred_world_mesh_path = os.path.join(dir, "%s.obj" % fn)
        pred_world_mesh = load_ori_mesh(pred_world_mesh_path)
        pred_align_mesh = pred_world_mesh.copy()

        # perspective to orthogonal
        pred_align_mesh.vertices *= (1015/112 / abs((np.mean(pred_world_mesh.vertices[:,2])-10)))

        # move to cropped image coordinate
        pred_align_mesh.vertices[:,:2] += 1
        pred_align_mesh.vertices = pred_align_mesh.vertices * 224/2 

        # compensate left up because of cropping
        pred_align_mesh.vertices[:,0] += left
        pred_align_mesh.vertices[:,1] -= up

        # align to original image
        pred_align_mesh.vertices /= scale
        pred_align_mesh.vertices[:,1] = gt_img_size - 1 - (round(224/scale) - 1 - \
                                        pred_align_mesh.vertices[:,1])

        # move to orthogonal coordinate
        pred_align_mesh.vertices /= (gt_img_size/2)
        pred_align_mesh.vertices[:,:2] -= 1

        if f_gt > 0:
            # orthogonal to perspective
            pred_align_mesh.vertices = R_pers2ortho.dot(pred_align_mesh.vertices.T).T
            pred_align_mesh.vertices[:, 2] -= np.mean(pred_align_mesh.vertices[:,2])
            pred_align_mesh.vertices[:, 2] += (f_gt / (gt_img_size/2))

            # scale to make face width = 1
            pred_align_mesh.vertices /= (np.max(pred_align_mesh.vertices[:, 1]) - \
                                         np.min(pred_align_mesh.vertices[:, 1]))/3   
        else:
            # keep orthogonal camera, add bias to make pyrender work
            pred_align_mesh.vertices[:,2] -= (np.mean(pred_align_mesh.vertices[:,2]) + render_bias)
    except:
        pred_align_mesh = None
    return pred_align_mesh


# zhuhao
def align_dfdn(dir, fn, f_gt=-1):

    try:
        # read mesh
        src_mesh = load_ori_mesh(dir + "%s.obj" % fn)
        
        # read camera parameters
        affine_mat = np.loadtxt(dir + "%s.affine_from_ortho.txt" % fn)
        view_mat =  np.loadtxt(dir + "%s.modelview.txt" % fn)
        scale = np.linalg.norm(affine_mat[0, :3])
        t = affine_mat[:2, 3]
        R = view_mat[:3, :3]

        pred_align_mesh = src_mesh.copy()

        # apply rotation
        pred_align_mesh.vertices = np.dot(R, pred_align_mesh.vertices.T).T

        # apply scale
        pred_align_mesh.vertices *= scale

        # apply translation
        pred_align_mesh.vertices[:, 1] *= -1
        pred_align_mesh.vertices[:, :2] += t

        # move to orthogonal cameara
        pred_align_mesh.vertices[:, :2] -= (gt_img_size/2)
        pred_align_mesh.vertices /= (gt_img_size/2)
        pred_align_mesh.vertices[:, 1] *= -1

        if f_gt > 0:
            # orthogonal to perspective
            pred_align_mesh.vertices = R_pers2ortho.dot(pred_align_mesh.vertices.T).T
            pred_align_mesh.vertices[:, 2] -= np.mean(pred_align_mesh.vertices[:,2])
            pred_align_mesh.vertices[:, 2] += (f_gt / (gt_img_size/2))
            
            # scale to make face width = 1
            pred_align_mesh.vertices /= (np.max(pred_align_mesh.vertices[:, 1]) - \
                                         np.min(pred_align_mesh.vertices[:, 1]))/3   
        else:
            # keep orthogonal camera, add bias to make pyrender work
            pred_align_mesh.vertices[:,2] -= (np.mean(pred_align_mesh.vertices[:,2]) + render_bias)
    
    except:
        pred_align_mesh = None
    return pred_align_mesh


# zhuhao
def align_lap(dir, fn, f_gt=-1):
    try:
        # read mesh
        pred_world_mesh_path = os.path.join(dir, "%s.obj" % fn)
        pred_world_mesh = load_ori_mesh(pred_world_mesh_path)
        pred_align_mesh = pred_world_mesh.copy()

        # read camera parameters
        with open(os.path.join(dir, "%s_cam.json" % fn), 'r') as f:
            cam_param = json.load(f)
        K = np.array(cam_param[0])
        Rt = np.array(cam_param[1])
        f = K[0, 0]
        crop_center = np.array(cam_param[2][:2])
        crop_scale = cam_param[2][2]

        # align to crop image
        pred_align_mesh.vertices -= np.array([0, 0, 1])
        pred_align_mesh.vertices = np.dot(Rt[:3, :3], pred_align_mesh.vertices.T).T
        pred_align_mesh.vertices += np.array([0, 0, 1])

        pred_align_mesh.vertices += (np.array([0, 0, -1]) - Rt[:, 3]) * 100

        pred_align_mesh.vertices[:, 0] *= -1
        pred_align_mesh.faces = pred_align_mesh.faces[:, [1, 0, 2]]

        # inverse scale by adding delta Z
        pred_align_mesh.vertices[:, 2] -= np.mean(pred_align_mesh.vertices[:, 2])*(1 - 1/crop_scale)

        # compensate cx cy
        f_new = f/128*gt_img_size
        mean_z = np.mean(pred_align_mesh.vertices[:, 2])
        pred_align_mesh.vertices[:, 0] += (128//2*(2-(crop_scale*gt_img_size)/128)/f_new*mean_z)
        pred_align_mesh.vertices[:, 1] -= (128//2*(2-(crop_scale*gt_img_size)/128)/f_new*mean_z)

        # inverse crop
        delta_R = (1 - crop_scale) * gt_img_size // 2
        pred_align_mesh.vertices[:, 0] += ((delta_R + crop_center[1] - 128)
                                          /f_new*np.abs(np.mean(pred_align_mesh.vertices[:, 2])))
        pred_align_mesh.vertices[:, 1] -= ((delta_R + crop_center[0] - 128)
                                          /f_new*np.abs(np.mean(pred_align_mesh.vertices[:, 2])))

        if f_gt > 0:
            # gl perspective camera to cv perspective camera
            pred_align_mesh.vertices = np.dot(R_pers2ortho, pred_align_mesh.vertices.T).T

            # scale to fit focal length
            mean_z = np.mean(pred_align_mesh.vertices[:,2])
            pred_align_mesh.vertices[:,2] -= mean_z
            pred_align_mesh.vertices *= (f*2/f_gt)
            pred_align_mesh.vertices[:,2] += mean_z

            # scale to make face width = 1
            pred_align_mesh.vertices /= (np.max(pred_align_mesh.vertices[:, 1]) - \
                                         np.min(pred_align_mesh.vertices[:, 1]))/3
        else:
            # gl perspective cam to orthogonal cam
            pred_align_mesh.vertices = np.dot(R_gl2ortho, pred_align_mesh.vertices.T).T

            # perspective to orthogonal
            pred_align_mesh.vertices *= (f / np.mean(pred_align_mesh.vertices[:, 2]) / 128 * 2)
            pred_align_mesh.vertices = R_pers2ortho.dot(pred_align_mesh.vertices.T).T

            # add bias to make pyrender work
            pred_align_mesh.vertices[:,2] -= (np.mean(pred_align_mesh.vertices[:,2]) + render_bias)
    except:
        pred_align_mesh = None
    return pred_align_mesh

# zhuhao
def align_rafare(dir, fn, f_gt=-1):
    
    # read mesh
    pred_world_mesh_path = os.path.join(dir, "%s.obj" % fn)
    pred_world_mesh = load_ori_mesh(pred_world_mesh_path)
    pred_align_mesh = pred_world_mesh.copy()
    
    # move to orthogonal camera coordinate
#    pred_align_mesh.vertices /= (gt_img_size/2)
#    pred_align_mesh.vertices[:, :2] -= 1
    
    if f_gt > 0:
        # orthogonal to perspective
        pred_align_mesh.vertices = R_pers2ortho.dot(pred_align_mesh.vertices.T).T
        pred_align_mesh.vertices[:, 2] -= np.mean(pred_align_mesh.vertices[:,2])        
        pred_align_mesh.vertices[:, 2] += (f_gt / (gt_img_size/2))
                
        # scale to make face width = 1
        pred_align_mesh.vertices /= (np.max(pred_align_mesh.vertices[:, 1]) - \
                                     np.min(pred_align_mesh.vertices[:, 1]))/3
    else:
        # keep orthogonal camera, add bias to make pyrender work
        pred_align_mesh.vertices[:,2] -= (np.mean(pred_align_mesh.vertices[:,2]) + render_bias)
    return pred_align_mesh


# ==================== dispatcher ====================
load_dispatcher = {'facescape_opti': align_facescape_opti,
                   'facescape_deep': align_facescape_deep, # OK
                   '3DDFA_V2': align_3ddfav2, # OK
                   'DECA': align_deca, # OK
                   'extreme3dface': align_ext3df,
                   'MGCNet': align_mgcnet, # OK
                   'PRNet': align_prnet, # OK
                   'RingNet': align_ringnet, # OK
                   'SADRNet': align_sadrnet, # OK
                   'UDL': align_udl, # OK
                   'DF2Net':align_df2net, # OK
                   'Deep3DFaceRec':align_deep3df, # OK
                   'DFDN': align_dfdn,
                   'LAP': align_lap, # OK
                   'RAFaRe': align_rafare,
                  }

# ==================== tool functions ====================

# used in align_mgcnet()
def rotMtx_eular(euler_tensor):
    phi = euler_tensor[0] # x
    theta = euler_tensor[1] # y
    psi = euler_tensor[2] # z

    s_ph = np.sin(phi) # x
    c_ph = np.cos(phi)

    s_t = np.sin(theta) # y
    c_t = np.cos(theta)

    s_ps = np.sin(psi) # z
    c_ps = np.cos(psi)
    rot = np.array([[c_t * c_ps, -c_t * s_ps, s_t], 
                    [c_ph * s_ps + c_ps * s_ph * s_t, c_ph * c_ps - s_ph * s_t * s_ps, -c_t * s_ph], 
                    [s_ps * s_ph - c_ph * c_ps * s_t, c_ps * s_ph + c_ph * s_t * s_ps, c_t * c_ph]])
    return rot

# used in align_udl()
def calc_trans_matrix(para_pose):
    pitch, yaw, roll, tx, ty, sth = para_pose
    
    cos_x, sin_x = np.cos(pitch), np.sin(pitch)
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    cos_z, sin_z = np.cos(roll), np.sin(roll)
    
    # compute rotation matrices
    rx = np.array([[1, 0, 0], 
                   [0, cos_x, sin_x], 
                   [0, -sin_x, cos_x]])
    
    ry = np.array([[cos_y, 0, -sin_y], 
                   [0, 1, 0], 
                   [sin_y, 0, cos_y]])
    
    rz = np.array([[cos_z, sin_z, 0], 
                   [-sin_z, cos_z, 0], 
                   [0, 0, 1]])

    scale_ratio = 1.5 # constant value to make face to be large in the image
    R = np.dot(np.dot(rx, ry), rz)
    R = R * np.exp(sth) * scale_ratio
    t = np.array([tx, ty, -200])
    return R, t
