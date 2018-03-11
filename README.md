# Style Aggregated Network for Facial Landmark Detection

We provide the training and testing codes for [SAN](https://d-x-y.github.io/publication/style-aggregation-network), implemented in [PyTorch](pytorch.org).

## Preparation

### Dependencies
- [Python3.6](https://www.anaconda.com/download/#linux)
- [PyTorch](http://pytorch.org/)
- [torchvision](http://pytorch.org/docs/master/torchvision)

### Datasets download
- Download 300W-Style and AFLW-Style from [Google Drive](https://drive.google.com/open?id=14f2lcJVF6E4kIICd8icUs8UuF3J0Mutd), and extract the downloaded files into `~/datasets/`.
- In 300W-Style and AFLW-Style directories, the `Original` sub-directory contains the original images from [300-W](https://ibug.doc.ic.ac.uk/resources/300-W/) and [AFLW](https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/aflw/)
<img src="cache_data/cache/dataset.jpg" width="480">
Figure 1. Our 300W-Style and AFLW-Style datasets. There are four styles, original, sketch, light, and gray.

# The Core Codes will come soon before 1st June.

### Generate lists for training and evaluation
```
cd cache_data
python aflw_from_mat.py
python generate_300W.py
```
The generated list file will be saved into `./cache_data/lists/300W` and `./cache_data/lists/AFLW`.

### Prepare images for the style-aggregated face generation module
```
python crop_pic.py
```
The above commands are used to pre-crop the face images.

## Training and Evaluation

### 300-W
- Step-1 : cluster images into different groups, for example `sh scripts/300W/300W_Cluster.sh 0,1 GTB 3`.
- Step-2 : use `sh scripts/300W/300W_CYCLE_128.sh 0,1 GTB` or `sh scripts/300W/300W_CYCLE_128.sh 0,1 DET` to train SAN on 300-W.

### AFLW
- Step-1 : cluster images into different groups, for example `sh scripts/AFLW/AFLW_Cluster.sh 0,1 GTB 3`.
- Step-2 : use `sh scripts/AFLW/AFLW_CYCLE_128.FULL.sh` or `sh scripts/AFLW/AFLW_CYCLE_128.FRONT.sh` to train SAN on AFLW.

## Citation
Please cite the following paper in your publications if it helps your research:
```
@inproceedings{dong2018san,
   title={Style Aggregated Network for Facial Landmark Detection},
   author={Dong, Xuanyi and Yan, Yan and Ouyang, Wanli and Yi, Yang},
   booktitle={Computer Vision and Pattern Recognition},
   year={2018},
}
```

## Contact
To ask questions or report issues, please open an issue on the [issues tracker](https://github.com/D-X-Y/SAN/issues).
