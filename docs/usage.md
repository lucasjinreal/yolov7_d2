## Training

You can refer to `install.md` for preparing your own dataset. Basically, just convert your dataset into coco format, and it's ready to go.


## Inference

You can direcly call `demo.py` to inference, visualize. A classic command would be:

```
python demo.py --config-file configs/coco/sparseinst/sparse_inst_r50vd_giam_aug.yaml --video-input ~/Movies/Videos/86277963_nb2-1-80.flv -c 0.4 --opts MODEL.WEIGHTS weights/sparse_inst_r50vd_giam_aug_8bc5b3.pth
```

## Deploy

YOLOv7 can be easily deploy via ONNX, you can using `export_onnx.py` and according config file to convert.

You u got any problems on any model arch, please fire an issue.
