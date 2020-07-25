## Extract Face


The [facial mask](/tools/extract_face/face_seg/png) in UV space is provided to extract the facial part from the full head TU-model. As shown in the figure below, there is the full head model in the right, and the extracted face in the left. 

<img src="/tools/extract_face/ef_result.jpg" width="600"> 

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
python extract_face.py
```
The result image is saved as ef_result.jpg.



