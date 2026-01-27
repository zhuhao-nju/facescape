# FaceScape

*FaceScape* provides large-scale high-quality 3D face datasets, parametric models, docs and toolkits about 3D face related technology. [[CVPR2020 paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_FaceScape_A_Large-Scale_High_Quality_3D_Face_Dataset_and_Detailed_CVPR_2020_paper.pdf) &nbsp;&nbsp;[[extended arXiv Report]](https://arxiv.org/pdf/2111.01082.pdf) &nbsp;&nbsp; [[supplementary]](https://openaccess.thecvf.com/content_CVPR_2020/supplemental/Yang_FaceScape_A_Large-Scale_CVPR_2020_supplemental.zip)

Our latest progress will be updated to this repository constantly - *[latest update: 2026/01/27]*

### Data

*New:* The data can be accessed at the new website https://nju-3dv.github.io/projects/FaceScape/. The old website (facescape.nju.edu.cn) will be decommissioned soon.

<img src="/figures/facescape_all.jpg" width="800">

The available sources include:

| Item (Docs)              | Description                                                         | Quantity                                         | Quality |
|-------------------|---------------------------------------------------------------------|------------------------------------------------|---------|
| [TU models](/doc/doc_tu_model.md) | Topologically uniformed 3D face models <br>with displacement map and texture map. | **16940 models** <br>(847 id × 20 exp)       |  Detailed geometry, <br>4K dp/tex maps |
| [Multi-view data](/doc/doc_mview_model.md) | Multi-view images, camera parameters <br>and corresponding 3D face mesh. | **>400k images** <br>(359 id × 20 exp <br>× ≈60 view)|  4M~12M pixels       |
| [Bilinear model](/doc/doc_bilinear_model.md) | The statistical model to transform the base <br>shape into the vector space.  |   4 for different settings      |    Only for base shape.    |
| [Info list](/doc/doc_tu_model.md)         | Gender / age of the subjects.                                        |   847 subjects   |    --    |

The datasets are only released for non-commercial research use.  As facial data involves the privacy of participants, we use strict license terms to ensure that the dataset is not abused.

### Benchmark for SVFR
We present a benchmark to evaluate the accuracy of single-view face 3D reconstruction (SVFR) methods, view [here](/benchmark/README.md) for the details.

### ToolKit
Start using python toolkit [here](/toolkit/README.md), the demos include:

* [bilinear_model-basic](https://nbviewer.jupyter.org/github/zhuhao-nju/facescape/blob/master/toolkit/demo_bilinear_basic.ipynb) - use facescape bilinear model to generate 3D mesh models.
* [bilinear_model-fit](https://nbviewer.jupyter.org/github/zhuhao-nju/facescape/blob/master/toolkit/demo_bilinear_fit.ipynb) - fit the bilinear model to 2D/3D landmarks.
* [multi-view-project](https://nbviewer.jupyter.org/github/zhuhao-nju/facescape/blob/master/toolkit/demo_mview_projection.ipynb) - Project 3D models to multi-view images.
* [landmark](https://nbviewer.jupyter.org/github/zhuhao-nju/facescape/blob/master/toolkit/demo_landmark.ipynb) - extract landmarks using predefined vertex index.
* [facial_mask](https://nbviewer.jupyter.org/github/zhuhao-nju/facescape/blob/master/toolkit/demo_mask.ipynb) - extract facial region from the full head TU-models.
* [render](https://nbviewer.jupyter.org/github/zhuhao-nju/facescape/blob/master/toolkit/demo_render.ipynb) - render TU-models to color images and depth map.
* [alignment](https://nbviewer.jupyter.org/github/zhuhao-nju/facescape/blob/master/toolkit/demo_align.ipynb) - align all the multi-view models.
* [symmetry](https://nbviewer.jupyter.org/github/zhuhao-nju/facescape/blob/master/toolkit/demo_symmetry.ipynb) - get the correspondence of the vertices on TU-models from left side to right side.
* [rig](https://nbviewer.jupyter.org/github/zhuhao-nju/facescape/blob/master/toolkit/demo_rig.ipynb) - rig 20 expressions to 52 expressions.


### Our More Projects related to FaceScape

**[Towards Native Generative Model for 3D Head Avatar (Fundamental Research 2026)]()**  
*Yiyu Zhuang\*, Hao Zhu\*, Jiawei Zhang\*, Yuxiao He\*, Yanwen Wang, Jiahe Zhu, Yao Yao, Siyu Zhu, Xun Cao<sup>#</sup>*

**[FATE: Full-head Gaussian Avatar with Textural Editing from Monocular Video (CVPR 2025)](https://zjwfufu.github.io/FATE-page/)**  
*Jiawei Zhang, Zijian Wu, Zhiyang Liang, Yicheng Gong, Dongfang Hu, Yao Yao, Xun Cao, Hao Zhu<sup>#</sup>*

**[DicFace: Dirichlet-Constrained Variational Codebook Learning for Temporally Coherent Video Face Restoration (CVPR 2025)](https://github.com/fudan-generative-vision/DicFace)**  
*Yan Chen\*, Hanlin Shang\*, Ce Liu, Yuxuan Chen, Hui Li, Weihao Yuan, Hao Zhu, Zilong Dong, Siyu Zhu<sup>#</sup>*

**[VividTalk: One-Shot Audio-Driven Talking Head Generation Based on 3D Hybrid Prior (3DV 2025)]()**  
*Xusen Sun, Longhao Zhang, Hao Zhu<sup>#</sup>, Peng Zhang<sup>#</sup>, Bang Zhang, Xinya Ji, Kangneng Zhou, Daiheng Gao, Liefeng Bo, Xun Cao*

**[Hallo2: Long-Duration and High-Resolution Audio-Driven Portrait Image Animation (ICLR 2025)](https://fudan-generative-vision.github.io/hallo2/#/)**  
*Jiahao Cui\*, Hui Li\*, Yao Yao, Hao Zhu, Hanlin Shang, Kaihui Cheng, Hang Zhou, Siyu Zhu<sup>#</sup>, Jingdong Wang*

**[EmoTalk3D: High-Fidelity Free-View Synthesis of Emotional 3D Talking Head (ECCV 2024)](https://nju-3dv.github.io/projects/EmoTalk3D)**  
*Qianyun He, Xinya Ji, Yicheng Gong, Yuanxun Lu, Zhengyu Diao, Linjia Huang, Yao Yao, Siyu Zhu, Zhan Ma, Songcen Xu, Xiaofei Wu, Zixiao Zhang, Xun Cao, Hao Zhu<sup>#</sup>*

**[Head360: Learning a Parametric 3D Full-Head for Free-View Synthesis in 360° (ECCV 2024)](https://nju-3dv.github.io/projects/Head360)**  
*Yuxiao He, Yiyu Zhuang, Yanwen Wang, Yao Yao, Siyu Zhu, Xiaoyu Li, Qi Zhang, Xun Cao, Hao Zhu<sup>#</sup>*

**[High-fidelity 3D Face Generation from Natural Language Descriptions (CVPR 2023)](https://github.com/zhuhao-nju/describe3d)**  
*Menghua Wu, Hao Zhu<sup>#</sup>, Linjia Huang, Yiyu Zhuang, Yuanxun Lu, Xun Cao*

**[RAFaRe: Learning Robust and Accurate Non-parametric 3D Face Reconstruction from Pseudo 2D&3D Pairs (AAAI 2023)](https://github.com/zhuhao-nju/rafare)**  
*Longwei Guo, Hao Zhu<sup>#</sup>, Yuanxun Lu, Menghua Wu, Xun Cao*

**[Detailed Facial Geometry Recovery from Multi-view Images by Learning an Implicit Function (AAAI 2022)](https://github.com/zhuhao-nju/mvfr)**  
*Yunze Xiao\*, Hao Zhu\*, Haotian Yang, Zhengyu Diao, Xiangju Lu, Xun Cao*


### ChangeLog
* **2026/01/27** <br>
The download website for the FaceScape dataset has been relocated to https://nju-3dv.github.io/projects/FaceScape/. All data can now be accessed on the new site.
* **2023/10/20** <br>
Benchmark data and results have been updated to be consistent with the experiments in the latest journal version paper.
* **2022/9/9** <br>
One section is added to introduce open-source projects that use FaceScape data or models, and will be continuously updated.
* **2022/7/26** <br>
The data for training and testing [MoFaNeRF](https://github.com/zhuhao-nju/mofanerf) is added to the [download page](https://facescape.nju.edu.cn/).
* **2021/12/2** <br>
A benchmark to evaluate single-view face reconstruction is available, view [here](https://github.com/zhuhao-nju/facescape/blob/master/benchmark/README.md) for detail.
* **2021/8/16** <br>
Share link on Google Drive is available after requesting the license key, view [here](https://github.com/zhuhao-nju/facescape/blob/master/doc/facescape_googledrive.md) for details.
* **2021/5/13** <br>
The fitting demo is added to the toolkit. Please note if you downloaded the bilinear model v1.6 before 2021/5/13, you need to download it again, because some parameters required by the fitting demo are supplemented.
* **2021/4/14** <br>
The bilinear model has been updated to 1.6, check it [here](/doc/doc_bilinear_model.md).<br>
The new bilinear model can now be downloaded from *NJU Drive* or *Google Drive* without requesting a license key. Check it [here](/doc/external_link_fsbm.md).<br>
ToolKit and Doc have been updated with new content.<br>
Some wrong ages and genders in the info list are corrected in "info_list_v2.txt".<br>
* **2020/9/27** <br>
The code of detailed riggable 3D face prediction is released, check it [here](https://github.com/yanght321/Detailed3DFace.git).<br>
* **2020/7/25** <br>
Multi-view data is available for download.<br>
The bilinear model is updated to ver 1.3, with vertex-color added.<br>
Info list including gender and age is available on the download page.<br>
Tools and samples are added to this repository.<br>
* **2020/7/7** <br>
The bilinear model is updated to ver 1.2.
* **2020/6/13** <br>
The [website]((https://facescape.nju.edu.cn/)) of FaceScape is online. <br>3D models and bilinear models are available for download.<br>
* **2020/3/31** <br>
The pre-print paper is available on [arXiv](https://arxiv.org/abs/2003.13989).<br>

### Bibtex
If you find this project helpful to your research, please consider citing:

```
@article{zhu2023facescape,
  title={FaceScape: 3D Facial Dataset and Benchmark for Single-View 3D Face Reconstruction},
  author={Zhu, Hao and Yang, Haotian and Guo, Longwei and Zhang, Yidi and Wang, Yanru and Huang, Mingkai and Wu, Menghua and Shen, Qiu and Yang, Ruigang and Cao, Xun},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
  year={2023},
  publisher={IEEE}}
```
```
@inproceedings{yang2020facescape,
  author = {Yang, Haotian and Zhu, Hao and Wang, Yanru and Huang, Mingkai and Shen, Qiu and Yang, Ruigang and Cao, Xun},
  title = {FaceScape: A Large-Scale High Quality 3D Face Dataset and Detailed Riggable 3D Face Prediction},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2020},
  page = {601--610}}
```

### Acknowledge
The project is supported by [CITE Lab](https://cite.nju.edu.cn/) of Nanjing University, Baidu Research, and Aiqiyi Inc.  The student contributors: Shengyu Ji, Wei Jin, Mingkai Huang, Yanru Wang, Haotian Yang, Yidi Zhang, Yunze Xiao, Yuxin Ding, Longwei Guo, Menghua Wu, Yiyu Zhuang.
