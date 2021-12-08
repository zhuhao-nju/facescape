## ================================================================================
## Description: save results to csv table
##
## Usage: 'python export_table.py' to export csv table to '../eval_result/*.csv'
##        csv table can be further transformed to Latex, Excel, etc.
##
## Authors: Hao Zhu (zhuhaoese@nju.edu.cn)
##
## License: MIT
## ================================================================================

import numpy as np

require_visi = False

dataset_list = ['fswild', 'fslab']
method_list = ['extreme3dface', # CVPR 2018
               'PRNet', # ECCV 2018
               'Deep3DFaceRec', # CVPRW 2019
               'RingNet', # CVPR 2019
               'DFDN', # ICCV 2019
               'DF2Net', # ICCV 2019
               'UDL', # TIP 2020
               'facescape_opti', # CVPR 2020
               'facescape_deep', # CVPR 2020
               'MGCNet', # ECCV 2020
               '3DDFA_V2', # ECCV 2020
               'SADRNet', # TIP 2021
               'LAP', # CVPR 2021
               'DECA', # SIGGRAPH 2021
              ]

metric_list = ['CD_full', 'MNE_full', 'CR_full']

# Table 1, fs_wild table - angle
dataset_name = 'fswild'
with open("../eval_result/table_1_fswild_angle.csv", 'w') as f:
    # table header
    f.write(",,$\pm5$,,,$\pm30$,,,$\pm60$,,,$\pm90$,,\n")
    f.write(",CD/$mm$,MNE/$rad$,CR/$\%$,CD/$mm$,MNE/$rad$,CR/$\%$,CD/$mm$,MNE/$rad$,CR/$\%$," + 
            "CD/$mm$,MNE/$rad$,CR/$\%$,done/$\%$\n")
    
    for method_name in method_list:
        f.write("%s," % method_name)

        value_arr = np.zeros((13))
        for m_i, metric_name in enumerate(metric_list):
            this_list = np.loadtxt("../eval_result/%s_%s/%s.txt" % (dataset_name, 
                                                                    method_name, 
                                                                    metric_name))

            angel_0_5 = np.mean(this_list[0:100][~np.isnan(this_list[0:100])])
            angel_5_30 = np.mean(this_list[100:200][~np.isnan(this_list[100:200])])
            angel_30_60 = np.mean(this_list[200:300][~np.isnan(this_list[200:300])])
            angel_60_90 = np.mean(this_list[300:][~np.isnan(this_list[300:])])
            angle_all = np.mean(this_list[~np.isnan(this_list)])
            done_rate = float(len(this_list[~np.isnan(this_list)])) / len(this_list)

            value_arr[0*3+m_i] = angel_0_5
            value_arr[1*3+m_i] = angel_5_30
            value_arr[2*3+m_i] = angel_30_60
            value_arr[3*3+m_i] = angel_60_90
            value_arr[12] = done_rate
                
        f.write("%1.2f,%1.3f,%2.1f," % (value_arr[0], value_arr[1], value_arr[2]*100))
        f.write("%1.2f,%1.3f,%2.1f," % (value_arr[3], value_arr[4], value_arr[5]*100))
        f.write("%1.2f,%1.3f,%2.1f," % (value_arr[6], value_arr[7], value_arr[8]*100))
        f.write("%1.2f,%1.3f,%2.1f," % (value_arr[9], value_arr[10], value_arr[11]*100))
        f.write("%2.1f\n" % (value_arr[12]*100))
        
        
# Table 2, fs_lab table - angle
dataset_name = 'fslab'

angle_0_indices, angle_30_indices, angle_60_indices = [], [], []
for sub_idx in range(20):
    for f_idx in range(1):
        for v_idx in [0]:
            angle_0_indices.append(sub_idx*33 + f_idx*11 + v_idx) 
        for v_idx in list(range(1, 9)):
            angle_30_indices.append(sub_idx*33 + f_idx*11 + v_idx) 
        for v_idx in list(range(9, 11)):
            angle_60_indices.append(sub_idx*33 + f_idx*11 + v_idx)
            
with open("../eval_result/table_2_fslab_angle.csv", 'w') as f:
    # table header
    f.write(",,$\pm5$,,,$\pm30$,,,$\pm60$,,\n")
    f.write(",CD/$mm$,MNE/$rad$,CR/$\%$,CD/$mm$,MNE/$rad$,CR/$\%$,CD/$mm$,MNE/$rad$,CR/$\%$," + 
            "done/$\%$\n")
    
    for method_name in method_list:
        f.write("%s," % method_name)
        
        value_arr = np.zeros((10))
        for m_i, metric_name in enumerate(metric_list):
            this_list = np.loadtxt("../eval_result/%s_%s/%s.txt" % (dataset_name, 
                                                                    method_name, 
                                                                    metric_name))

            angel_0 = np.mean(this_list[angle_0_indices][~np.isnan(this_list[angle_0_indices])])
            angel_30 = np.mean(this_list[angle_30_indices][~np.isnan(this_list[angle_30_indices])])
            if len(this_list[angle_60_indices][~np.isnan(this_list[angle_60_indices])]) == 0:
                angel_60 = -1
            else:
                angel_60 = np.mean(this_list[angle_60_indices][~np.isnan(this_list[angle_60_indices])])
            angle_all = np.mean(this_list[~np.isnan(this_list)])
            done_rate = float(len(this_list[~np.isnan(this_list)])) / len(this_list)
            
            value_arr[0*3+m_i] = angel_0
            value_arr[1*3+m_i] = angel_30
            value_arr[2*3+m_i] = angel_60
            value_arr[9] = done_rate
                
        f.write("%1.2f,%1.3f,%2.1f," % (value_arr[0], value_arr[1], value_arr[2]*100))
        f.write("%1.2f,%1.3f,%2.1f," % (value_arr[3], value_arr[4], value_arr[5]*100))
        f.write("%1.2f,%1.3f,%2.1f," % (value_arr[6], value_arr[7], value_arr[8]*100))
        f.write("%2.1f\n" % (value_arr[9]*100))
        

# Table 3, fs_lab table - focal length
dataset_name = 'fslab'

f_1200_indices, f_600_indices, f_300_indices = [], [], []
for sub_idx in range(20):
    for v_idx in range(11):
        f_1200_indices.append(sub_idx*33 + 0*11 + v_idx)
        f_600_indices.append(sub_idx*33 + 1*11 + v_idx)
        f_300_indices.append(sub_idx*33 + 2*11 + v_idx)

with open("../eval_result/table_3_fslab_f.csv", 'w') as f:
    # table header
    f.write(",,$1200$,,,$600$,,,$300$,,\n")
    f.write(",CD/$mm$,MNE/$rad$,CR/$\%$,CD/$mm$,MNE/$rad$,CR/$\%$,CD/$mm$,MNE/$rad$,CR/$\%$," + 
            "done/$\%$\n")
    
    for method_name in method_list:
        f.write("%s," % method_name)
        
        value_arr = np.zeros((10))
        for m_i, metric_name in enumerate(metric_list):
            this_list = np.loadtxt("../eval_result/%s_%s/%s.txt" % (dataset_name, 
                                                                    method_name, 
                                                                    metric_name))
            
            f_1200 = np.mean(this_list[f_1200_indices][~np.isnan(this_list[f_1200_indices])])
            f_600 = np.mean(this_list[f_600_indices][~np.isnan(this_list[f_600_indices])])
            f_300 = np.mean(this_list[f_300_indices][~np.isnan(this_list[f_300_indices])])
            f_all = np.mean(this_list[~np.isnan(this_list)])
            done_rate = float(len(this_list[~np.isnan(this_list)])) / len(this_list)
            
            value_arr[0*3+m_i] = f_1200
            value_arr[1*3+m_i] = f_600
            value_arr[2*3+m_i] = f_300
            value_arr[9] = done_rate
            
        f.write("%1.2f,%1.3f,%2.1f," % (value_arr[0], value_arr[1], value_arr[2]*100))
        f.write("%1.2f,%1.3f,%2.1f," % (value_arr[3], value_arr[4], value_arr[5]*100))
        f.write("%1.2f,%1.3f,%2.1f," % (value_arr[6], value_arr[7], value_arr[8]*100))
        f.write("%2.1f\n" % (value_arr[9]*100))
    
