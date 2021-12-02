## ================================================================================
## Description: evaluate methods on fs_wild or fs_lab dataset
##
## Arguments: dataset_name   - string, one of ['fswild', 'fslab']. 
##            method_name    - string, one of ['facescape_opti', 'facescape_deep', 
##                             '3DDFA_V2', 'DECA', 'extreme3dface', 'MGCNet', 
##                             'PRNet', 'RingNet', 'SADRNet', 'UDL', 'DF2Net', 
##                             'Deep3DFaceRec', 'fswild_Deep3DFaceRec']. Can be 
##                              extended by adding functions in pred_loader.py.
##            save_heat_mesh - bool, weather to save out intermediate meshes.
##
## Usage: see main() function in the end.
##
## Authors: Hao Zhu (zhuhaoese@nju.edu.cn), Longwei Guo, Menghua Wu
##
## License: MIT
## ================================================================================

import numpy as np, pickle, os, cv2, tqdm, argparse, json
from mesh_util import compute_chamfer, load_ori_mesh, make_trimesh, depth2mesh, cylinder2mesh
from renderer import render_orthcam, render_cvcam
from scipy.optimize import minimize
from pred_loader import load_dispatcher

class fs_evaluator():
    def __init__(self, dataset_name="fswild", method_name="facescape_opti", save_heat_mesh = True):
        
        self.dataset = dataset_name
        self.method = method_name
        self.save_heat_mesh = save_heat_mesh
        self.gt_mesh_dir = "../data/%s_gt_mesh/" % self.dataset[2:]
        self.pred_mesh_dir = "../pred/%s_pred/%s/" % (self.dataset[2:], self.method)
        self.save_dir = "../eval_result/%s_%s/" % (self.dataset, self.method)
        self.rend_size = 256
        self.ortho_scale = 128
        self.std_pupil_dist = (64.0 + 61.7) / 2 # mean pupil distance of male and female
        self.lab_subject_num = 20
        self.lab_fl_num = 3
        self.lab_view_num = 11
            
        if self.dataset == "fswild":
            self.data_num = 400
        elif self.dataset == "fslab":
            self.data_num = self.lab_subject_num * self.lab_fl_num * self.lab_view_num
        
        # result list
        self.cd_full_list = [] # chamfer distance (full)
        self.cr_full_list = [] # complete rate (full)
        self.mne_full_list = [] # normal mean error rate (full)
        self.pupil_scale_list = []
        
        if self.dataset == "fswild":
            
            # read pupil_scale, head_center and rotation
            self.pupil_scale_list = np.load("./predef/wild_pupil_scales.npy")
            self.head_center_list = np.load("./predef/wild_head_centers.npy")
            self.gt_R_list = np.load("./predef/wild_rotations.npy")

        elif self.dataset == "fslab":

            # read head_center and rotation
            self.pupil_scale_list = None #not required, as it has been applied in ground-truth mesh
            self.head_center_list = np.load("./predef/lab_head_centers.npy")
            self.gt_R_list = np.load("./predef/lab_rotations.npy")
                
        else:
            print("ERROR: dataset name should be 'fslab' or 'fswild', but get %s" % self.dataset)
            return False
        
        
    def run(self, iter_num = -1):
        if self.dataset == "fswild":
            self.eval_fswild(iter_num = iter_num if (iter_num>0 and iter_num<=400) else 400)
        elif self.dataset == "fslab":
            self.eval_fslab(iter_num = iter_num if (iter_num>0 and iter_num<=660) else 660)
            
    def parse_head_mesh(self, idx, eye_index):
        
        head_mesh_dir = "../benchmark_data_wild/gt_head_mesh/"
        
        # read gt head mesh
        gt_head_mesh_world = load_ori_mesh(os.path.join(head_mesh_dir, "%03d" % idx, "%03d.obj" % idx))
        
        # compute pupil distance
        mean_left_eye = np.mean(gt_head_mesh_world.vertices[eye_index[0]], 0)
        mean_right_eye = np.mean(gt_head_mesh_world.vertices[eye_index[1]], 0)
        mesh_pupil_dist = np.linalg.norm(mean_left_eye - mean_right_eye)
        pupil_scale = self.std_pupil_dist / mesh_pupil_dist
        
        # compute head center
        head_center = np.mean(gt_head_mesh_world.vertices, 0)
        
        return pupil_scale, head_center
    
    
    def optimize_Z(self, gt_world_mesh, pred_align_mesh, pupil_scale, K = None):
        
        if K is None: # orthogonal projection
            # render depth_gt
            depth_gt, _ = render_orthcam(gt_world_mesh, xy_mag = (pupil_scale, pupil_scale), 
                                         rend_size = (self.rend_size, self.rend_size))
        
            # render depth_pred
            depth_pred, _ = render_orthcam(pred_align_mesh, xy_mag = (pupil_scale, pupil_scale), 
                                           rend_size = (self.rend_size, self.rend_size))        
        else: # perspective projection
            # render depth_gt
            depth_gt, _ = render_cvcam(gt_world_mesh, K = K, 
                                       rend_size = (self.rend_size, self.rend_size))
            # render depth_pred
            depth_pred, _ = render_cvcam(pred_align_mesh, K = K, 
                                         rend_size = (self.rend_size, self.rend_size))        

        # get intersection mask
        mask = (depth_gt != 0) * (depth_pred != 0)
        depth_gt_area = np.sum(depth_gt!=0)
        
        depth_gt = depth_gt * mask
        depth_pred = depth_pred * mask
        
        if K is None: # orthogonal projection
            depth_gt *= (self.ortho_scale / pupil_scale)
            depth_pred *= (self.ortho_scale / pupil_scale)
            
        # optimize for z
        def optimize_z(z, depth_a, depth_b, mask_ab):
            error = np.sum(np.abs(depth_a - z - depth_b)[mask_ab])
            return error
        
        delta_z = minimize(optimize_z, 0, args = (depth_gt, depth_pred, mask), method='SLSQP').x[0]
        return delta_z, depth_gt, depth_pred, mask, depth_gt_area

    def compute_mne_full(self, gt_normal, pred_normal, gt_loc, pred_loc):
        
        cyl_gt_face_normal = np.zeros((self.rend_size, self.rend_size, 3))
        cyl_pred_face_normal = np.zeros((self.rend_size, self.rend_size, 3))
        
        gt_loc_x = np.array([para // self.rend_size for para in gt_loc[1]])
        gt_loc_y = np.array([para % self.rend_size for para in gt_loc[1]])
        cyl_gt_face_normal[gt_loc_x, gt_loc_y, :] = gt_normal[gt_loc[2]]
        
        pred_loc_x = np.array([para // self.rend_size for para in pred_loc[1]])
        pred_loc_y = np.array([para % self.rend_size for para in pred_loc[1]])
        cyl_pred_face_normal[pred_loc_x, pred_loc_y, :] = pred_normal[pred_loc[2]]
        
        normal_mask_full = np.bitwise_and((np.sum((cyl_gt_face_normal!=0), 2) > 0), 
                                          (np.sum((cyl_pred_face_normal!=0), 2) > 0))    
        normal_mask_full_c3 = np.stack((normal_mask_full, )*3, 2) # make 3-channel mask
        
        cyl_gt_face_normal = (cyl_gt_face_normal + 1) / np.linalg.norm(cyl_gt_face_normal + 1, 
                                                                       axis = 2, keepdims = True)
        cyl_pred_face_normal = (cyl_pred_face_normal + 1) / np.linalg.norm(cyl_pred_face_normal + 1, 
                                                                           axis = 2, keepdims = True)
        
        cyl_gt_face_normal = cyl_gt_face_normal * normal_mask_full_c3
        cyl_pred_face_normal = cyl_pred_face_normal * normal_mask_full_c3
        
        mne_full_map = np.arccos(np.sum(cyl_gt_face_normal * cyl_pred_face_normal, 2))
        
        mne_full = np.sum(mne_full_map[normal_mask_full]) / np.sum(normal_mask_full)
        
        return mne_full


    def compute_mne_visi(self, cyl_mesh_gt, cyl_mesh_pred, norm_scale):
        # render visible normal map
        cyl_mesh_gt.vertices[:, 2] = cyl_mesh_gt.vertices[:, 2] - 100
        cyl_mesh_pred.vertices[:, 2] = cyl_mesh_pred.vertices[:, 2] - 100
        
        normal_gt = render_orthcam(cyl_mesh_gt, 
                                   xy_mag = (norm_scale, norm_scale), 
                                   rend_size = (self.rend_size, self.rend_size), 
                                   smooth=False)[1].astype(np.int32)
        
        normal_pred = render_orthcam(cyl_mesh_pred, 
                                     xy_mag = (norm_scale, norm_scale), 
                                     rend_size = (self.rend_size, self.rend_size), 
                                     smooth=False)[1].astype(np.int32)
        
        # get intersection
        normal_mask_visi = np.bitwise_and(np.sum((normal_gt==255), 2)==0, 
                                      np.sum((normal_pred==255), 2)==0)
        normal_mask_visi_c3 = np.stack((normal_mask_visi, )*3, 2) # make 3-channel mask
        
        normal_gt = normal_gt * normal_mask_visi_c3
        normal_pred = normal_pred * normal_mask_visi_c3
        
        # compute mne
        with np.errstate(divide='ignore', invalid='ignore'):
            normal_gt = normal_gt / np.linalg.norm(normal_gt, axis = 2, keepdims = True)
            normal_pred = normal_pred / np.linalg.norm(normal_pred, axis = 2, keepdims = True)

            mne_visi_map = np.arccos(np.sum(normal_gt * normal_pred, 2))
        
        mne_visi = np.sum(mne_visi_map[normal_mask_visi]) / np.sum(normal_mask_visi)
        
        return mne_visi
    
    def map_cyl(self, gt_world_mesh, pred_align_mesh, R, head_center, pupil_scale):
        
        N = self.rend_size
        
        # make ray directions
        indices_arr = np.indices((N, N))
        angle_arr = indices_arr[1] * 2 * np.pi / 255.
        point_arr = np.stack((-np.sin(angle_arr), np.zeros((N, N)), -np.cos(angle_arr)), 2)
        ray_directions = point_arr.reshape(N*N , 3)
        
        # make ray origins
        ray_origins = np.zeros((N, N, 3))
        ray_origins[:, :, 1] = np.linspace(150, -150, N)
        ray_origins = ray_origins.transpose((1, 0, 2)).reshape((N*N, 3))
        
        # will remove in the future version
        std_cyl_gt_mesh = gt_world_mesh.copy()
        std_cyl_gt_mesh.vertices = std_cyl_gt_mesh.vertices - head_center * pupil_scale
        std_cyl_gt_mesh.vertices = np.tensordot(R.T, std_cyl_gt_mesh.vertices.T, 1).T
        
        gt_loc = std_cyl_gt_mesh.ray.intersects_location(ray_origins, 
                                                         ray_directions, 
                                                         multiple_hits=True)
            
        std_cyl_pred_mesh = pred_align_mesh.copy()
        std_cyl_pred_mesh.vertices = std_cyl_pred_mesh.vertices - head_center * pupil_scale
        std_cyl_pred_mesh.vertices = np.tensordot(R.T, std_cyl_pred_mesh.vertices.T, 1).T
        
        pred_loc = std_cyl_pred_mesh.ray.intersects_location(ray_origins, 
                                                             ray_directions, 
                                                             multiple_hits=True)
        
        dpmap_gt = np.zeros((N, N), dtype=np.float64)
        gt_loc0_norm = np.linalg.norm(gt_loc[0][:, [0, 2]], axis = 1)
        for ind in range(len(gt_loc[1])):
            dpmap_gt[gt_loc[1][ind] // N][gt_loc[1][ind] % N] = gt_loc0_norm[ind]
        
        dpmap_pred = np.zeros((N, N), dtype=np.float64)
        pred_loc0_norm = np.linalg.norm(pred_loc[0][:, [0, 2]], axis = 1)
        for ind in range(len(pred_loc[1])):
            dpmap_pred[pred_loc[1][ind] // N][pred_loc[1][ind] % N] = pred_loc0_norm[ind]        
        
        ## save to cyl
        cyl_mask = (dpmap_gt != 0) * (dpmap_pred != 0)
        
        return std_cyl_gt_mesh, std_cyl_pred_mesh, gt_loc, pred_loc, dpmap_gt, dpmap_pred, cyl_mask

    def make_color_bar(self, length = 256, width = 20):
        color_bar_value = np.stack((list(range(length-1, -1, -1)),) * width, 1).astype(np.uint8)
        self.color_bar = cv2.applyColorMap(color_bar_value, cv2.COLORMAP_JET)
        return True
    
    def make_heatmesh(self, src_mesh, vert_error, max_value=None, min_value=0):
        if max_value == None:
            max_value = np.max(vert_error)
        else:
            max_value = float(max_value)
        
        # replace NaN with max value for rendering
        vert_error[np.isnan(vert_error)] = max_value
        
        vert_error[vert_error < min_value] = min_value
        vert_error[vert_error > max_value] = max_value
        error_posi = ((vert_error - min_value) / (max_value - min_value) * \
                      (len(self.color_bar) - 1)).astype(np.int)
        vert_colors = self.color_bar[:,0,:][error_posi]
        
        tgt_mesh = make_trimesh(src_mesh.vertices, 
                                src_mesh.faces, 
                                vert_colors = vert_colors)
        return tgt_mesh
    
    def str2tri(self, name):
        id_exp, f, v = name.split('.')[0].split('_')
        return (int(id_exp), int(f), int(v))
    
    def tri2str(self, tri):
        return "%02d_%01d_%02d" % (tri[0], tri[1], tri[2])
    
    def index_tri2mono(self, tri):
        return (tri[0]*(self.lab_fl_num * self.lab_view_num) + tri[1] * self.lab_view_num + tri[2])
    
    def index_mono2tri(self, ind):
        id_exp = ind // (self.lab_fl_num * self.lab_view_num)
        f = (ind % (self.lab_fl_num * self.lab_view_num)) // self.lab_view_num
        v = (ind % (self.lab_fl_num * self.lab_view_num)) % self.lab_view_num
        return (id_exp, f, v)
    
    
    def eval_fswild(self, iter_num = 400):
        
        if iter_num < 0:
            iter_num = self.data_num
        elif iter_num > self.data_num:
            print("warning: number to evaluate exceeds data_num: %d > %d." % (iter_num, self.data_num), 
                  "set it as data_num.")
        
        for idx in tqdm.trange(iter_num):
            
            file_name = str(idx).zfill(3)
            
            # read gt mesh
            # ==================================================
            gt_world_mesh_path = os.path.join(self.gt_mesh_dir, file_name + ".obj")
            gt_world_mesh = load_ori_mesh(gt_world_mesh_path)
            
            # read predicted mesh and align in X and Y
            # ==================================================
            pred_align_mesh = load_dispatcher[self.method](self.pred_mesh_dir, file_name, -1)
            
            if pred_align_mesh is None:
                print("result of %03d not found, all metrics are set to NaN." % idx)
                self.cd_full_list.append(np.nan)
                self.cr_full_list.append(np.nan)
                self.mne_full_list.append(np.nan)
                continue
            
            # normalized scale using pupil distance
            # ==================================================
            pupil_scale = self.pupil_scale_list[idx]
            gt_world_mesh.vertices *= pupil_scale
            pred_align_mesh.vertices *= pupil_scale
            
            # align in depth direction in Z by optimizing detla_z
            # ==================================================
            delta_z, depth_gt, depth_pred, mask, depth_gt_area = self.optimize_Z(gt_world_mesh, 
                                                                                 pred_align_mesh, 
                                                                                 pupil_scale)
            
            # apply delta_z to pred_mesh
            pred_align_mesh.vertices[:, 2] -= delta_z / self.ortho_scale * pupil_scale
            
            # project to cyl coordinate
            # ==================================================
            std_cyl_gt_mesh, std_cyl_pred_mesh, gt_loc, pred_loc, dpmap_gt, dpmap_pred, cyl_mask \
                = self.map_cyl(gt_world_mesh, pred_align_mesh, 
                               self.gt_R_list[idx], 
                               self.head_center_list[idx], 
                               pupil_scale)
            
            # compute Compeleteness Rate
            # ==================================================
            self.cr_full_list.append(np.sum(cyl_mask) / np.sum(dpmap_gt != 0))

            # compute Chanmfer Distance (full)
            # ==================================================
            cyl_mesh_gt = cylinder2mesh(dpmap_gt * cyl_mask)  # cyl2mesh
            cyl_mesh_pred = cylinder2mesh(dpmap_pred * cyl_mask)
            
            cyl_mesh_gt.vertices = np.tensordot(self.gt_R_list[idx], cyl_mesh_gt.vertices.T, 1).T
            cyl_mesh_pred.vertices = np.tensordot(self.gt_R_list[idx], cyl_mesh_pred.vertices.T, 1).T
            
            cd_full, error_pred, _ = compute_chamfer(cyl_mesh_gt, cyl_mesh_pred, require_array=True)
            self.cd_full_list.append(cd_full)
            
            # compute Normal Mean Error (full)
            # ==================================================
            self.mne_full_list.append(self.compute_mne_full(std_cyl_gt_mesh.face_normals, 
                                                            std_cyl_pred_mesh.face_normals, 
                                                            gt_loc, pred_loc))
            
            # generate heat map for perspective projected mesh (full region)
            # ==================================================
            if self.save_heat_mesh is True:
                self.make_color_bar()
                cyl_mesh_pred.faces = cyl_mesh_pred.faces[:, [1, 0, 2]]
                heatmesh_visi = self.make_heatmesh(cyl_mesh_pred, error_pred, 
                                                   max_value = 5)

                os.makedirs(self.save_dir + '/heatmesh/', exist_ok=True)
                heatmesh_visi.export(self.save_dir + '/heatmesh/' + file_name + '.obj')
            
            print("CD = %.3f" % self.cd_full_list[-1], end="\t")
            print("CR = %.3f" % self.cr_full_list[-1], end="\t")
            print("MNE = %.3f" % self.mne_full_list[-1])
        
        # save results
        os.makedirs(self.save_dir, exist_ok = True)
        np.savetxt(self.save_dir + "CD_full.txt", self.cd_full_list)
        np.savetxt(self.save_dir + "CR_full.txt", self.cr_full_list)
        np.savetxt(self.save_dir + "MNE_full.txt", self.mne_full_list)
        
    
    def eval_fslab(self, iter_num = 660):
        
        if iter_num < 0:
            iter_num = self.data_num
        elif iter_num > self.data_num:
            print("warning: number to evaluate exceeds data_num: %d > %d." % (iter_num, self.data_num), 
                  "set it as data_num.")
        
        for mono_idx in tqdm.trange(iter_num):
            
            tri_idx = self.index_mono2tri(mono_idx)
            file_name = self.tri2str(tri_idx)
            
            # read gt mesh
            # ==================================================
            gt_world_mesh_path = os.path.join(self.gt_mesh_dir, "%02d.obj" % tri_idx[0])
            gt_world_mesh = load_ori_mesh(gt_world_mesh_path)
            gt_world_mesh.vertices *= self.std_pupil_dist # to world scale
            
            # read cam parameters
            with open("../data/lab_cam/%s.json" % file_name, 'r') as f:
                cam_param = json.load(f)
            K = np.array(cam_param[0])
            Rt = np.array(cam_param[1])
            Rt[:, 3] *= self.std_pupil_dist
            f_gt = K[0, 0]
            
            # apply Rt
            gt_world_mesh.vertices = Rt[:3,:3].dot(gt_world_mesh.vertices.T).T + Rt[:, 3]
            
            # read predicted mesh and align in X and Y
            # ==================================================
            pred_align_mesh = load_dispatcher[self.method](self.pred_mesh_dir, file_name, f_gt)
            
            
            if pred_align_mesh is None:
                print("result of %s not found, all metrics are set to NaN." % file_name)
                self.cd_full_list.append(np.nan)
                self.cr_full_list.append(np.nan)
                self.mne_full_list.append(np.nan)
                continue
            
            pred_align_mesh.vertices *= self.std_pupil_dist
            
            # align in depth direction in Z by optimizing detla_z
            # ==================================================
            delta_z, depth_gt, depth_pred, mask, depth_gt_area = self.optimize_Z(gt_world_mesh, 
                                                                                 pred_align_mesh, 
                                                                                 pupil_scale = 1, 
                                                                                 K = K)
            
            # apply delta_z to pred_mesh
            mean_z = np.mean(depth_gt[mask])
            pred_align_mesh.vertices *= (mean_z/(mean_z-delta_z))
        
            # project to cyl coordinate
            # ==================================================
            std_cyl_gt_mesh, std_cyl_pred_mesh, gt_loc, pred_loc, dpmap_gt, dpmap_pred, cyl_mask \
                = self.map_cyl(gt_world_mesh, pred_align_mesh, 
                               self.gt_R_list[tri_idx[2]], 
                               self.head_center_list[tri_idx[1]], 
                               pupil_scale = 1)
            
            # compute Compeleteness Rate
            # ==================================================
            self.cr_full_list.append(np.sum(cyl_mask) / np.sum(dpmap_gt != 0))
            
            # compute Chanmfer Distance (full)
            # ==================================================
            cyl_mesh_gt = cylinder2mesh(dpmap_gt * cyl_mask)  # cyl2mesh
            cyl_mesh_pred = cylinder2mesh(dpmap_pred * cyl_mask)
            
            cyl_mesh_gt.vertices = np.tensordot(self.gt_R_list[tri_idx[2]], 
                                                cyl_mesh_gt.vertices.T, 1).T
            cyl_mesh_pred.vertices = np.tensordot(self.gt_R_list[tri_idx[2]], 
                                                  cyl_mesh_pred.vertices.T, 1).T
            
            cd_full, error_pred, _ = compute_chamfer(cyl_mesh_gt, cyl_mesh_pred, require_array=True)
            self.cd_full_list.append(cd_full)
        
            # compute Normal Mean Error (full)
            # ==================================================
            self.mne_full_list.append(self.compute_mne_full(std_cyl_gt_mesh.face_normals, 
                                                            std_cyl_pred_mesh.face_normals, 
                                                            gt_loc, pred_loc))
            
            # generate heat map for perspective projected mesh (full region)
            # ==================================================
            if self.save_heat_mesh is True:
                self.make_color_bar()
                cyl_mesh_pred.faces = cyl_mesh_pred.faces[:, [1, 0, 2]]
                heatmesh_visi = self.make_heatmesh(cyl_mesh_pred, error_pred, 
                                                   max_value = 5)
                
                os.makedirs(self.save_dir + '/heatmesh/', exist_ok=True)
                heatmesh_visi.export(self.save_dir + '/heatmesh/' + file_name + '.obj')
            
            print("CD = %.3f" % self.cd_full_list[-1], end="\t")
            print("CR = %.3f" % self.cr_full_list[-1], end="\t")
            print("MNE = %.3f" % self.mne_full_list[-1])
        
        # save results
        os.makedirs(self.save_dir, exist_ok = True)
        np.savetxt(self.save_dir + "CD_full.txt", self.cd_full_list)
        np.savetxt(self.save_dir + "CR_full.txt", self.cr_full_list)
        np.savetxt(self.save_dir + "MNE_full.txt", self.mne_full_list)
            
            
def main():
    parser = argparse.ArgumentParser(description='Evaluator of FaceScape single-view recon benchmark.')
    parser.add_argument('--dataset', type=str, help="fswild or fslab.")
    parser.add_argument('--method', type=str, help="method name to be evaluated.")
    parser.add_argument('--num', type=int, default = -1, help="numbers to process.")
    parser.add_argument('--heat_mesh', type=bool, default=True, help="save heat mesh or not.")
    
    args = parser.parse_args()
    
    fs_eval = fs_evaluator(dataset_name = args.dataset, 
                           method_name = args.method, 
                           save_heat_mesh = args.heat_mesh)
    fs_eval.run(args.num)

if __name__ == "__main__":
    main()
