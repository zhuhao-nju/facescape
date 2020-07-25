## Facescape Bilinear Model

Our bilinear model is a statistical model which transforms the base shape of the faces into a vector space representation. We provide two 3DMM with different numbers of identity parameters:
 
 - **core_847_50_52.npy** - bilinear model with 52 expression parameters and 50 identity parameters.
- **core_847_300_52.npy** - bilinear model with 52 expression parameters and 300 identity parameters.
- **factors_id_847_50_52.npy** and factors_id_847_300_52.npy are identity parameters corresponding to 847 subjects in the dataset.


### Usage
The demo code is provided to briefly explain the usage of our model.  The demo has been testd in Python 3.x.  Numpy and Pickle libraries are required.  Please firstly download **'facescape_bilinear_model_v1_3.zip'** from the download page, then extract the 'data' folder to the current directory, and finally run:

```
python demo_bilinear.py
```
The demo will export a head mesh model and a facial colored mesh model, which are generated from the given parameters.

Please refer to our [paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_FaceScape_A_Large-Scale_High_Quality_3D_Face_Dataset_and_Detailed_CVPR_2020_paper.pdf) for more about the bilinear model. 

### Version Changes

 - **v1.3** - Current version. Bilinear model with vertex color is supplemented.
 - **v1.2** - The previous v1.0 (core_50_52.npy) is build from a different trainset of 888 subjects.  This version uses the index consistent with the index of released TU-models.
 - **v1.0** - Initial release.
