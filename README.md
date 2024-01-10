# Engilish
*  **Theory** : [https://wikidocs.net/226341](https://wikidocs.net/226341) <br>
*  **Implementation** : [https://wikidocs.net/226342](https://wikidocs.net/226342)

# 한글
*  **Theory** : [https://wikidocs.net/225898](https://wikidocs.net/225898) <br>
*  **Implementation** : [https://wikidocs.net/226040](https://wikidocs.net/226040)

This repository is folked from [https://github.com/yjh0410/RT-ODLab](https://github.com/yjh0410/RT-ODLab).
At this repository, simplification and explanation and will be tested at Colab Environment.

# YOLOv8:

|   Model   |  Batch | Scale | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | FLOPs<br><sup>(G) | Params<br><sup>(M) | Weight |
|-----------|--------|-------|------------------------|-------------------|-------------------|--------------------|--------|
| YOLOv8-N  | 8xb16  |  640  |          36.8          |        52.9       |        8.8        |         3.2        | [ckpt](https://github.com/yjh0410/RT-ODLab/releases/download/yolo_tutorial_ckpt/yolov8_n_coco.pth) |
| YOLOv8-S  | 8xb16  |  640  |                        |                   |                   |                    |  |
| YOLOv8-M  | 8xb16  |  640  |                        |                   |                   |                    |  |
| YOLOv8-L  | 8xb16  |  640  |          50.2          |        68.0       |       165.7       |         43.7       | [ckpt](https://github.com/yjh0410/RT-ODLab/releases/download/yolo_tutorial_ckpt/yolov8_l_coco.pth) |

- For training, we train YOLOv8 series with 500 epochs on COCO.
- For data augmentation, we use the large scale jitter (LSJ), Mosaic augmentation and Mixup augmentation, following the setting of [YOLOv8](https://github.com/ultralytics/yolov8).
- For optimizer, we use AdamW with weight decay 0.05 and base per image lr 0.001 / 64, which is different from the official YOLOv8. We have tried SGD, but it has weakened performance. For example, when using SGD, YOLOv8-N's AP was only 35.8%, lower than the current result (36.8 %), perhaps because some hyperparameters were not set properly.
- For learning rate scheduler, we use linear decay scheduler.

## Step 1. Clone from Github and install library

Git clone to root directory. 

```Shell
# Clone from Github Repository
! git init .
! git remote add origin https://github.com/RichardMinsooGo-ML/Bible_4_Part_F_08_Pytorch_Yolov8.git
# ! git pull origin master
! git pull origin main
```

A tool to count the FLOPs of PyTorch model.

```
from IPython.display import clear_output
clear_output()
```

```Shell
! pip install thop
```

## Step x. Download pretrained weight

```Shell
! wget https://github.com/yjh0410/RT-ODLab/releases/download/yolo_tutorial_ckpt/yolov8_n_coco.pth
! wget https://github.com/yjh0410/RT-ODLab/releases/download/yolo_tutorial_ckpt/yolov8_l_coco.pth
```

## Demo
### Detect with Image
```Shell
# Detect with Image

! python demo.py --mode image \
                 --path_to_img /content/dataset/demo/images/ \
                 --cuda \
                 -m yolov8_l \
                 --weight /content/yolov8_l_coco.pth \
                 -size 640 \
                 -vt 0.4 \
                 # --show

# See /content/det_results/demos/image
```

### Detect with Video
```Shell
# Detect with Video

! python demo.py --mode video \
                 --path_to_vid /content/dataset/demo/videos/street.mp4 \
                 --cuda \
                 -m yolov8_l \
                 --weight /content/yolov8_l_coco.pth \
                 -size 640 \
                 -vt 0.4 \
                 --gif
                 # --show

# See /content/det_results/demos/video Download and check the results
```

### Detect with Camera
```Shell
# Detect with Camera
# it don't work at Colab. Use laptop

# ! python demo.py --mode camera \
#                  --cuda \
#                  -m yolov8_l \
#                  --weight /content/yolov8_l_coco.pth \
#                  -size 640 \
#                  -vt 0.4 \
#                  --gif
                 # --show
```

## Download COCO Dataset

```Shell
# COCO dataset download and extract

# ! wget http://images.cocodataset.org/zips/train2017.zip
! wget http://images.cocodataset.org/zips/val2017.zip
! wget http://images.cocodataset.org/zips/test2017.zip
# ! wget http://images.cocodataset.org/zips/unlabeled2017.zip

# ! unzip train2017.zip  -d dataset/COCO
! unzip val2017.zip  -d dataset/COCO
! unzip test2017.zip  -d dataset/COCO

# ! unzip unlabeled2017.zip -d dataset/COCO

# ! rm train2017.zip
# ! rm val2017.zip
# ! rm test2017.zip
# ! rm unlabeled2017.zip

! wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
# wget http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip
# wget http://images.cocodataset.org/annotations/image_info_test2017.zip
# wget http://images.cocodataset.org/annotations/image_info_unlabeled2017.zip

! unzip annotations_trainval2017.zip -d dataset/COCO
# ! unzip stuff_annotations_trainval2017.zip
# ! unzip image_info_test2017.zip
# ! unzip image_info_unlabeled2017.zip

# ! rm annotations_trainval2017.zip
# ! rm stuff_annotations_trainval2017.zip
# ! rm image_info_test2017.zip
# ! rm image_info_unlabeled2017.zip

clear_output()
```

## Test YOLOv8
Taking testing YOLOv8 on COCO-val as the example,
```Shell
# Test YOLOv8

# See /content/det_results/coco/yolov1

! python test.py --cuda \
                 -d coco \
                 --data_path /content/dataset \
                 -m yolov8_l \
                 --weight /content/yolov8_l_coco.pth \
                 -size 640 \
                 -vt 0.4
                 # --show
```

## Evaluate YOLOv8
Taking evaluating YOLOv8 on COCO-val as the example,
```Shell
# Evaluate YOLOv8

! python eval.py --cuda \
                 -d coco-val \
                 --data_path /content/dataset \
                 -m yolov8_l \
                 --weight /content/yolov8_l_coco.pth
```

# Training test
## Download VOC Dataset

```Shell
# VOC 2012 Dataset Download and extract

! wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
!tar -xvf "/content/VOCtrainval_11-May-2012.tar" -C "/content/dataset"
clear_output()

# VOC 2007 Dataset Download and extract

! wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
! wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
!tar -xvf "/content/VOCtrainval_06-Nov-2007.tar" -C "/content/dataset"
!tar -xvf "/content/VOCtest_06-Nov-2007.tar" -C "/content/dataset"
clear_output()
```


## Train YOLOv8
### Single GPU
Taking training YOLOv8_n on COCO as the example,
```Shell
! python train.py --cuda \
                  -d voc \
                  --data_path /content/dataset \
                  -m yolov8_n \
                  -bs 16 \
                  --max_epoch 5 \
                  --wp_epoch 1 \
                  --eval_epoch 5 \
                  --fp16 \
                  --ema \
                  --multi_scale
```
```
! python train.py --cuda \
                  -d voc \
                  --data_path /content/dataset \
                  -m yolov8_s \
                  -bs 16 \
                  --max_epoch 5 \
                  --wp_epoch 1 \
                  --eval_epoch 5 \
                  --fp16 \
                  --ema \
                  --multi_scale
```

```
! python train.py --cuda \
                  -d voc \
                  --data_path /content/dataset \
                  -m yolov8_m \
                  -bs 16 \
                  --max_epoch 5 \
                  --wp_epoch 1 \
                  --eval_epoch 5 \
                  --fp16 \
                  --ema \
                  --multi_scale
```

```
! python train.py --cuda \
                  -d voc \
                  --data_path /content/dataset \
                  -m yolov8_l \
                  -bs 8 \
                  --max_epoch 5 \
                  --wp_epoch 1 \
                  --eval_epoch 5 \
                  --fp16 \
                  --ema \
                  --multi_scale
```

```
# T4 GPU 14G
! python train.py --cuda \
                  -d voc \
                  --data_path /content/dataset \
                  -m yolov8_x \
                  -bs 8 \
                  --max_epoch 5 \
                  --wp_epoch 1 \
                  --eval_epoch 5 \
                  --fp16 \
                  --ema \
                  --multi_scale
```

### Multi GPU
Taking training YOLOv8 on COCO as the example,
```Shell
# Cannot test at Colab-Pro + environment

# ! python -m torch.distributed.run --nproc_per_node=8 train.py \
#                                   --cuda \
#                                   -dist \
#                                   -d voc \
#                                   --data_path /content/dataset \
#                                   -m yolov8_s \
#                                   -bs 128 \
#                                   -size 640 \
#                                   --wp_epoch 3 \
#                                   --max_epoch 300 \
#                                   --eval_epoch 10 \
#                                   --no_aug_epoch 20 \
#                                   --ema \
#                                   --fp16 \
#                                   --sybn \
#                                   --multi_scale \
#                                   --save_folder weights/
```


