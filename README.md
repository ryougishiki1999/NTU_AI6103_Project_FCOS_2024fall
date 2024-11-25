# [AI6103] Deep Learning: Fully Convolutional One-Stage Object Detection

## How to run

platform: linux-64
NAME="Ubuntu"
VERSION="20.04.3 LTS (Focal Fossa)"
5.4.0-153-generic

### Install the required packages
Install the required packages using the following command:
```
conda env create -f environment.yml
```

### Download the dataset

#### Download the Coco 2017 dataset
Run the Downloader Script provided in the scripts folder. This will download the dataset and extract it to the `data/` folder.
```
python scripts/coco_dataset_downloader.py
```
Final `data` folder structure should look like this:
```
data/
    coco_2017/
        raw/
        train/
        ...
```

#### Download the VOC2007 and VOC2012 dataset:
Download from the following links:

http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar

Then extract the files to the `data/` folder. Final folder structure should look like this:
```
data/
    VOCdevkit/
        VOC2007/
            Annotations/
            ImageSets/
            JPEGImages/
            ...
        VOC2012/
            Annotations/
            ImageSets/
            JPEGImages/
            ...
```
### Run the Scripts

**all the runnable scripts are under `scripts/` folder**

you can change the backbone in `src/config.py`, modifying `choosen_backbone`. **The default backbone is ResNet18 for Voc**
```python
choosen_backbone = Backbone.RestNet50 # change backbone here
```
Additionally, you can change parameters like `score_threshold` etc. in `src/config.py` to modify the training and evaluation process.


At present, we have following backbones:
- ResNet50: supports CoCo
- ResNet18: support Voc
- MobileNetV2: support Voc 
- VovNet27Slim: support Voc

#### CoCo Dataset

`coco_detect.py`:
detection script for coco dataset on backbone: ResNet50
use images under directory `assets/coco_detect_imagaes/` as input
the corresponding object detection results will be saved in `out/coco_detect_out_images/`

`coco_eval.py`:
evalution script for coco dataset on backbone: ResNet50

`coco_train.py`:
training script for coco dataset on backbone: ResNet50

#### Voc Dataset

`voc_detect.py`:
detection script for coco dataset on backbone: ResNet18, MobileNetV2, VovNet27Slim
use images under directory `assets/voc_detect_imagaes/` as input
the corresponding object detection results will be saved in `out/voc_detect_out_images/`

`voc_eval.py`:
evalution script for coco dataset on backbone: ResNet18, MobileNetV2, VovNet27Slim

`voc_train.py`:
training script for coco dataset on backbone: ResNet18, MobileNetV2, VovNet27Slim
