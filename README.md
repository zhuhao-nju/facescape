# FaceScape

This is the project page for our paper 
"FaceScape: a Large-Scale High Quality 3D Face Dataset and Detailed Riggable 3D Face Prediction". 
[[CVPR2020 paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_FaceScape_A_Large-Scale_High_Quality_3D_Face_Dataset_and_Detailed_CVPR_2020_paper.pdf) &nbsp;&nbsp; [[supplemetary]](https://openaccess.thecvf.com/content_CVPR_2020/supplemental/Yang_FaceScape_A_Large-Scale_CVPR_2020_supplemental.zip)

We will also update latest progress and available sources to this repository~ **[latest update: 2020/9/27]**

```diff
- We are very sorry that due to the renovation of the campus network, the FaceScape website (https://facescape.nju.edu.cn/)
- will not be accessible from October 1 to October 8, 2020.
```

### Dataset

The datasets are released in website: https://facescape.nju.edu.cn/.  

<img src="/figures/facescape_all.jpg" width="800"> 

The available sources include:

| Item              | Description                                                         | Quantity                                         | Quality |
|-------------------|---------------------------------------------------------------------|------------------------------------------------|---------|
| TU models | Topologically uniformed 3D face models <br>with displacement map and texture map. | **16940 models** <br>(847 id × 20 exp)       |  Detailed geometry, <br>4K dp/tex maps |
| Multi-view data | Multi-view images, camera paramters <br>and coresponding 3D face mesh. | **>400k images** <br>(359 id × 20 exp <br>× ≈60 view)|  4M~12M pixels       |
| Bilinear model | The statistical model to transform the base <br>shape into the vector space.  |   4 for different settings      |    Only for base shape.    |
| Info list         | Gender / age of the subjects.                                        |   847 subjects   |    --    |
| Tools |  Python code to generate **depth map**, <br>**landmarks**, **facial segmentation**, etc. |    --                                              |    --    |
 
 
The datasets are only released for non-commercial research use.  As facial data involves the privacy of participants, we use strict license terms to ensure that the dataset is not abused.  Please visit the [website](https://facescape.nju.edu.cn/) for more information. 


### Tools
 - [mview](/tools/mview/README.md) - parse and test multi-view images and corresponding 3D models.
 - [bilinear model](/tools/bilinear_model/README.md) - simple demo to use facescape bilinear model.
 - [landmark](/tools/landmark/README.md) - extract landmarks using predefined vertex index.
 - [extract face](/tools/extract_face/README.md) - extract facial region from the mesh of full head.
<!-- [render](/tools/render/README.md) - simple demo to render models to color image and depth map using pyrender. -->

### Code
The code of detailed riggable 3D face prediction in our paper is released [here](https://github.com/yanght321/Detailed3DFace.git).

### ChangeLog
* **2020/9/27** <br>
The code of detailed riggable 3D face prediction is released, check it [here](https://github.com/yanght321/Detailed3DFace.git).<br>
* **2020/7/25** <br>
Multi-view data is available for download, check it [here](/tools/mview/README.md).<br>
Bilinear model with vertex-color has been added to v1.3, check it [here](/tools/bilinear_model/README.md). <br>
Info list including gender and age is available in download page.<br>
Tools and samples are added to this repository.<br>
* **2020/7/7** <br>
Bilinear model v1.2 is updated, check it [here](/tools/bilinear_model/README.md).<br>
* **2020/6/13** <br>
The [website]((https://facescape.nju.edu.cn/)) of FaceScape is online. <br>3D models and bilinear models are available for download.<br>
* **2020/3/31** <br>
The pre-print paper is available on [arXiv](https://arxiv.org/abs/2003.13989).<br>

### Bibtex
```
@InProceedings{yang2020facescape,
  author = {Yang, Haotian and Zhu, Hao and Wang, Yanru and Huang, Mingkai and Shen, Qiu and Yang, Ruigang and Cao, Xun},
  title = {FaceScape: A Large-Scale High Quality 3D Face Dataset and Detailed Riggable 3D Face Prediction},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2020},
  page = {601--610}}
```
