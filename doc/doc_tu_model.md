# Facescape TU model

The data available for downloading contains 847 subjects × 20 expressions, in a total of 16940 models, which is roughly 90% of the complete data. The other 10% of data are not released for potential evaluation or benchmark in the future.

<img src="/figures/facescape_tu.jpg" width="600"> 

### Models
There are 847 tuple of topologically uniformed models. Each tuple of data consists of:

* 20 base mesh models (/models_reg/$IDENTITY$_$EXPRESSION$.obj)
* 20 displacement maps (/dpmap/$IDENTITY$_$EXPRESSION$.png)
* 1 base material file (/models_reg/$IDENTITY$_$EXPRESSION$.obj.mtl)
* 1 texture (/models_reg/$IDENTITY$_$EXPRESSION$.jpg) where $IDENTITY$ is the index of identity (1 - 847), $EXPRESSION$ is the index of expression (0 - 20). Please note that some of the model‘s texture maps (index: 360 - 847) were mosaics around the eyes to protect the privacy of some participants.

### Feature
* Topologically uniformed.
The geometric models of different identities and different expressions share the same mesh topology, which makes the features on faces easy to be aligned. This also helps in building a 3D morphable model.
* Displacement map + base mesh.
We use base shapes to represent rough geometry and displacement maps to represent detailed geometry, which is a two-layer representation for our extremely detailed face shape. Some light-weight software like MeshLab can only visualize the base mesh model/texture. Displacement maps can be loaded and visualized in MAYA, ZBrush, 3D MAX, etc.
* 20 specific expressions.
The subjects are asked to perform 20 specific expressions for capturing: neutral, smile, mouth-stretch, anger, jaw-left, jaw-right, jaw-forward, mouth-left, mouth-right, dimpler, chin-raiser, lip-puckerer, lip-funneler, sadness, lip-roll, grin, cheek-blowing, eye-closed, brow-raiser, brow-lower.
* High resolution.
The texture maps and displacement maps reach 4K resolution, which preserving maximum detailed texture and geometry.

<img src="/figures/facescape_info.jpg" width="600"> 

### Information list
A text file containing the ages and gender of the subjects. From left to right, each row is the index, gender (m-male, f-female), age, and valid label. ‘-’ means this information is not provided. Valid label is [1 + 4] binary number, 1-True, 0-False. The first number means if the model for this person is complete and valid, and the rest four means if obj-model, mtl-material, jpg-texture, and png-dpmap are missing.

### Publishable list
A text file containing the indexes of the model that can be used for paper publication or presentation. Please read the 4th term in the license for more about this policy. The publishable list may be updated in the future.

