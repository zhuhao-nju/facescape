## Landmark
For topology-uniformed model, the landmarks on the 3D mesh can be traced by pre-defined indeces which is saved in [landmark_indices.txt](/tools/landmark/landmark_indices.txt).  The definition of landmarks is shown in the figure below.

<img src="/tools/landmark/lm_result.jpg" width="600"> 

A simple demo is provided to generate this figure.  The demo is tested in Python 3.6 on Ubuntu.  Requirements:
```
sudo apt install cmake
pip install numpy pyrender openmesh opencv-python pillow
```
Run the following script in terminal to unzip the sample of the tu-model:
```
sudo apt-get install unzip
unzip ../../samples/sample_tu_model.zip -d ../../samples/sample_tu_model/
```
Run the demo:
```
python show_landmark.py
```
The result image is saved as lm_result.jpg.



