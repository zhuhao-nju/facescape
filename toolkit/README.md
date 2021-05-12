# README - toolkit

### Environment

The toolkit demo have been tested in python 3.6 in Ubuntu.  We recommend to create an environment using [Anaconda](https://www.anaconda.com/products/individual#Downloads):

```
conda create -n facescape python=3.6 -y
conda activate facescape
```

Install required packages
```
pip install -r requirements.txt
conda install jupyter -y # for running in local jupyter
conda install nb_conda_kernels -y # enable notebook kernels
```

Then you can view demos locally by starting jupter notebook:
```
jupyter notebook
```

### Download sample data

Run the following script to download sample data:
```
cd samples/ && ./download_sample.sh
```

### Demos

The links below use an external jupyter renderer, because github often fails to render jupyter notebook online. 

* [bilinear_model-basic](https://nbviewer.jupyter.org/github/zhuhao-nju/facescape/blob/master/toolkit/demo_bilinear_basic.ipynb) - use facescape bilinear model to generate 3D mesh models.
* [bilinear_model-fit](https://nbviewer.jupyter.org/github/zhuhao-nju/facescape/blob/master/toolkit/demo_bilinear_fit.ipynb) - fit the bilinear model to 2D/3D landmarks.
* [multi-view-project](https://nbviewer.jupyter.org/github/zhuhao-nju/facescape/blob/master/toolkit/demo_mview_projection.ipynb) - Project 3D models to multi-view images.
* [landmark](https://nbviewer.jupyter.org/github/zhuhao-nju/facescape/blob/master/toolkit/demo_landmark.ipynb) - extract landmarks using predefined vertex index.
* [facial_mask](https://nbviewer.jupyter.org/github/zhuhao-nju/facescape/blob/master/toolkit/demo_mask.ipynb) - extract facial region from the full head TU-models.
* [render](https://nbviewer.jupyter.org/github/zhuhao-nju/facescape/blob/master/toolkit/demo_render.ipynb) - render TU-models to color images and depth map. 
* [alignment](https://nbviewer.jupyter.org/github/zhuhao-nju/facescape/blob/master/toolkit/demo_align.ipynb) - align all the multi-view models.
* [symmetry](https://nbviewer.jupyter.org/github/zhuhao-nju/facescape/blob/master/toolkit/demo_symmetry.ipynb) - get the correspondence of the vertices on TU-models from left side to right side.
* rigging (Coming soon) - animate the TU-models to an simple animation with changing experssions.
 


