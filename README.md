# The Source Codes will come soon before 1st June.

# [Style Aggregated Network for Facial Landmark Detection](https://arxiv.org/abs/1803.04108)

We provide the training and testing codes for [SAN](https://d-x-y.github.io/publication/style-aggregation-network), implemented in [PyTorch](pytorch.org).

## Preparation

### Dependencies
- [Python3.6](https://www.anaconda.com/download/#linux)
- [PyTorch](http://pytorch.org/)
- [torchvision](http://pytorch.org/docs/master/torchvision)

### Datasets download
- Download 300W-Style and AFLW-Style from [Google Drive](https://drive.google.com/open?id=14f2lcJVF6E4kIICd8icUs8UuF3J0Mutd), and extract the downloaded files into `~/datasets/`.
- In 300W-Style and AFLW-Style directories, the `Original` sub-directory contains the original images from [300-W](https://ibug.doc.ic.ac.uk/resources/300-W/) and [AFLW](https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/aflw/).
- The sketch, light, and gray style images are used to analyze the image style variance in facial landmark detection.
<img src="cache_data/cache/dataset.jpg" width="480">
Figure 1. Our 300W-Style and AFLW-Style datasets. There are four styles, original, sketch, light, and gray.

##### 300W-Style Directory
- `300W-Gray` : 300W  afw  helen  ibug  lfpw
- `300W-Light` : 300W  afw  helen  ibug  lfpw
- `300W-Sketch` : 300W  afw  helen  ibug  lfpw
- `300W-Original` : 300W  afw  helen  ibug  lfpw
- `Bounding_Boxes`

##### AFLW-Style Directory
- `aflw-Gray` : 0 2 3
- `aflw-Light` : 0 2 3
- `aflw-Sketch` : 0 2 3
- `aflw-Original` : 0 2 3
- `annotation`: 0 2 3


### Generate lists for training and evaluation
```
cd cache_data
python aflw_from_mat.py
python generate_300W.py
```
The generated list file will be saved into `./cache_data/lists/300W` and `./cache_data/lists/AFLW`.

### Prepare images for training the style-aggregated face generation module
```
python crop_pic.py
```
The above commands will pre-crop the face images, and save them into `./cache_data/cache/300W` and `./cache_data/cache/AFLW`.


## Training and Evaluation

### 300-W
- Step-1 : cluster images into different groups, for example `sh scripts/300W/300W_Cluster.sh 0,1 GTB 3`.
- Step-2 : use `sh scripts/300W/300W_CYCLE_128.sh 0,1 GTB` or `sh scripts/300W/300W_CYCLE_128.sh 0,1 DET` to train SAN on 300-W.

### AFLW
- Step-1 : cluster images into different groups, for example `sh scripts/AFLW/AFLW_Cluster.sh 0,1 GTB 3`.
- Step-2 : use `sh scripts/AFLW/AFLW_CYCLE_128.FULL.sh` or `sh scripts/AFLW/AFLW_CYCLE_128.FRONT.sh` to train SAN on AFLW.

#### Normalization

<img src="cache_data/cache/figure_1_68.jpg" width="480">
Figure 2. We use the distance between the outer corners of the eyes, i.e., the 37-th and the 46-th points, for normalization.

## Citation
Please cite the following paper in your publications if it helps your research:
```
@inproceedings{dong2018san,
   title={Style Aggregated Network for Facial Landmark Detection},
   author={Dong, Xuanyi and Yan, Yan and Ouyang, Wanli and Yang, Yi},
   booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
   year={2018},
}
```

## Contact
To ask questions or report issues, please open an issue on the [issues tracker](https://github.com/D-X-Y/SAN/issues).
