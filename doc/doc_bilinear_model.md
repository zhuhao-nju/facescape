# Facescape Bilinear Model

Please check the version of bilinear model and refer to the corresponding description. 

<img src="/figures/facescape_bm.jpg" width="600"> 

### Description (ver1.6)

Comparing to previous versions, ver1.6 has updated in the following aspects :

* Resolve the problem of wierd fitting around ears.
* New model with better representation ability for frontal face is provided.
* New symmetric mesh topology is used.
* All parameters are integrated into a npz file and a python class is provided to use them.

Ver1.6 provides four models:

* *facescape_bm_v1.6_847_50_52_id_front.npz* - Bilinear model with 52 expression parameters and 50 identity parameters. PCA is applied to identity dimesion, which reduces from 847 to 50.  The frontal facial vertices are with higher weights (10:1) for a better representation ablility.  *This model is **recommended** in general cases.*
* *facescape_bm_v1.6_847_50_52_id.npz* - Same to above, except the higher frontal weights are not adopts.
* *facescape_bm_v1.6_847_300_52_id.npz* - Same to above, except the identity dimesion is reduced to 300, not 50.
* *facescape_bm_v1.6_847_50_52_id_exp.npz* - Same to above, except that PCA is applied to both identity dimension(50) and expression dimension(52). Please note that this model doesn't work for our code of 'bilinear_model-fit'.

The demo code to use the facescape bilinear model ver1.6 can be found here: [basic usage](https://nbviewer.jupyter.org/github/zhuhao-nju/facescape/blob/master/toolkit/demo_bilinear_basic.ipynb) and [model fitting](https://nbviewer.jupyter.org/github/zhuhao-nju/facescape/blob/master/toolkit/demo_bilinear_fit.ipynb). 

Please refer to our [paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_FaceScape_A_Large-Scale_High_Quality_3D_Face_Dataset_and_Detailed_CVPR_2020_paper.pdf) for more about the bilinear model. 


### Description (ver1.0/1.2/1.3)

Our bilinear model is a statistical model which transforms the base shape of the faces into a vector space representation. We provide two 3DMM with different numbers of identity parameters:
 
 - *core_847_50_52.npy* - bilinear model with 52 expression parameters and 50 identity parameters.
- *core_847_300_52.npy* - bilinear model with 52 expression parameters and 300 identity parameters.
- *factors_id_847_50_52.npy* and factors_id_847_300_52.npy are identity parameters corresponding to 847 subjects in the dataset.

The demo code to use the facescape bilinear model ver1.0/1.2/1.3 can be found [here](https://nbviewer.jupyter.org/github/zhuhao-nju/facescape/blob/master/toolkit/demo_bilinear_basic.ipynb).

### Version Log

 - **v1.6** - The topolpogy of mesh has been updated to be fully symmetric.  The facial mask is refined. The problem that may produce wierd fitting result around ears has been solved. The parameters required by fitting demo are also attached. (2021/12/08: The wrong id_mean, id_var, and the missing ft_indices_front have been fixed.)
 - **v1.3** - Bilinear model with vertex color is supplemented.
 - **v1.2** - The previous v1.0 (core_50_52.npy) is build from a different trainset of 888 subjects.  This version uses the index consistent with the index of released TU-models.
 - **v1.0** - Initial release.
