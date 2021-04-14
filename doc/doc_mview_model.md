## FaceScape - Multi-View

FaceScape provides multi-view images, camera paramters and reconstructed 3D shapes.  There are 359 subjects x 20 expressions = 7120 tuples of data.  The number of available images reaches to over 400k!  Please visit [https://facescape.nju.edu.cn/](https://facescape.nju.edu.cn/) for data access.  Users who have already applied FaceScape can login and download multi-view data directly.

<img src="/figures/facescape_mview.jpg" width="600"> 

### Data Structure
The multi-view images are stored as:
```
IMGS_ROOT
#     ↓ID    ↓EXP     ↓VIEW
├───── 1
│      ├── 1_neutral
│      │       ├───── 0.jpg
│      │       ├───── 1.jpg
│      │       ├───── ...
│      │       ├───── N.jpg
│      │       └───── params.json
│      ├── 2_smile
│      ├── ...             
│      └── 20_brow_lower   
├───── 2
├───── ...
└───── 359
```
The corresponding 3D models are stored as:
```
MESH_ROOT
#     ↓ID    ↓EXP
├───── 1
│      ├── 1_neutral.ply
│      ├── 2_smile.ply
│      ├── ...             
│      └── 20_brow_lower.ply
├───── 2
├───── ...
└───── 359
```
The 'params.json' files store the multiple information about the data.  The dictionary read from 'params.json' is organized as:

 - '$VIEW$_K' - Intrinsic Matrix [4x4 float]
 - '$VIEW$_Rt' - Extrinsic Matrix [3x4 float]
 - '$VIEW$_distortion' - Distortion Parameters (k1 k2 p1 p2 k3) [5 float]
 - '$VIEW$_width' - Image Width [int]
 - '$VIEW$_height' - Image Height [int]
 - '$VIEW$_matches' - Number of valid matches [int]
 - '$VIEW$_valid' - Image is valid or not [bool]
 - '$VIEW$_sn' - Serial Number of Camera [string]
 - '$VIEW$_ori' - Original Filename [string]
 
 where $VIEW$ is view index in the range of [0-N]. N is the image number of this tuple. 
 
### Implementation Notes
 
 - **Distortion**: To project the reconstructed model to fit the images, the images must be undistorted in advance with the provided distortion parameters (see Projection Test below).
 - **Valid_label**: The camera parameters of which the '$VIEW$_valid' label is False are not reliable. So when reconstructing with multi-view images, please ignore the images with False label in '$VIEW$_valid'.
 - **Publishing limit**: All the images and models in FaceScape cannot be shown in publications except for the subjects listed in the 'publishable_list' (available in download page after login).  This has been stated in the license agreement.
 - **Camera Model**: Please see Camera Model section below.

### Parameter Parser
A simple demo code is provided to parse the data.  Json package is required to be installed:
```
Pip install json
```
Parameters can be parsed by the following code in Python:

```python
import json

with open("img/$id$/$exp$.json", 'r') as f:
    params = json.load(f) # read parameters
img_num = len(params)//9 # get image number

test_view = 0 # select a view in 0 ~ img_num

K = params['%d_K' % test_view] # intrinsic mat
Rt = params['%d_Rt' % test_view] # extrinsic mat
dist = params['%d_distortion' % test_view] # distortion parameter
h_src = params['%d_height' % test_view] # height
w_src = params['%d_width' % test_view] # width
valid = params['%d_valid' % test_view] # valid or not
```


### Camera Model

The camera parameters of FaceScape multi-view data use the camera model that is widely used in computer vision field (CV-Cam for short).  This camera coordinate is different from the camera coordinate defined in OpenGL (GL-Cam for short) and many other rendering tools.  CV-Cam and GL-Cam are shown in the figure below.  

<img src="/figures/cam_coord_fs.jpg" width="600"> 

Our extrinsic matrix (Rt) can be transformed from CV-Cam to GL-Cam by the following calculation:
```python
import numpy as np
Rt_gl = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).dot(Rt_cv)
```

### Projection Test
A simple demo is provided to render the mesh model to fit the image.  The code can be found [here](https://nbviewer.jupyter.org/github/zhuhao-nju/facescape/blob/master/toolkit/demo_mview_projection.ipynb).
