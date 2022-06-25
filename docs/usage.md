## Training

You can refer to `install.md` for preparing your own dataset. Basically, just convert your dataset into coco format, and it's ready to go.

We have 3 **key** train scripts, they are:

- `train_coco.py`: this is basically most common used train script for coco;
- `train_detr.py`: use this for **any** DETR or transformer based model;
- `train_net.py`: Experimented changing training strategy script, **used for experiement**;
- `train_custom_datasets.py`: train all customized datasets;

For demo usage, you can using:

- `demo.py`: for demo visualize result;
- `demo_lazyconfig.py`: for demo using `*.py` as config file;


## Inference

You can direcly call `demo.py` to inference, visualize. A classic command would be:

```
python demo.py --config-file configs/coco/sparseinst/sparse_inst_r50vd_giam_aug.yaml --video-input ~/Movies/Videos/86277963_nb2-1-80.flv -c 0.4 --opts MODEL.WEIGHTS weights/sparse_inst_r50vd_giam_aug_8bc5b3.pth
```

## Deploy

YOLOv7 can be easily deploy via ONNX, you can using `export_onnx.py` and according config file to convert.

You u got any problems on any model arch, please fire an issue.
