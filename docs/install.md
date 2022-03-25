# YOLOv7 Install

> Please install detectron2 first, this is the basic dependency. For detectron2 just clone official repo and install follow their instructions.

yolov7 is not a lib, it's a project ready for use. But to install dependencies, there still need some process. 

First, please consider install 2 important lib that you might not familliar with:

```
alfred-py
nbnb
```

Both of them can be installed from pip. The first one provides enhanced and full-featured visualization utils for drawing boxes, masks etc. And it provides some very convenient tools for users to visialization your coco dataset (VOC, YOLO format also supported). After install, you can call `alfred` to get more details.

`nbnb` is a lib that provides some useful common network blocks.

Also, if you need fbnetv3, you need install mobilecv from FaceBook:

```
pip install git+https://github.com/facebookresearch/mobile-vision.git
```

After install, you can now ready to train with YOLOv7.

```
python train_net.py --config-file configs/coco/darknet53.yaml --num-gpus 8
```

train YOLOX:

```
python train_net.py --config-file configs/coco/yolox_s.yaml --num-gpus 8
```

## Train on Custom dataset

If you want train on custom dataset, you **just need convert your dataset to coco format**. And that's all, that's all you need do.

Then you just need create a new folder of your dataset under `configs`, and set your data path in config, take VisDrone dataset as example:

```
DATASETS:
  TRAIN: ("visdrone_train",)
  TEST: ("visdrone_val",)
```

Then register your dataset in `train_visdrone.py`:

```
DATASET_ROOT = './datasets/visdrone'
ANN_ROOT = os.path.join(DATASET_ROOT, 'visdrone_coco_anno')
TRAIN_PATH = os.path.join(DATASET_ROOT, 'VisDrone2019-DET-train/images')
VAL_PATH = os.path.join(DATASET_ROOT, 'VisDrone2019-DET-val/images')
TRAIN_JSON = os.path.join(ANN_ROOT, 'VisDrone2019-DET_train_coco.json')
VAL_JSON = os.path.join(ANN_ROOT, 'VisDrone2019-DET_val_coco.json')

register_coco_instances("visdrone_train", {}, TRAIN_JSON, TRAIN_PATH)
register_coco_instances("visdrone_val", {}, VAL_JSON, VAL_PATH)
```

Here, you set your json path, your images path, then you are ready to go.

