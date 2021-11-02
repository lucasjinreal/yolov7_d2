- 2021.11.02:

  this is Just a testing..
  

- 2021.08.30:
  
  We now get another AP of YOLOX without preprocessing:

  ```
  |   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
  |:------:|:------:|:------:|:------:|:------:|:------:|
  | 38.053 | 57.665 | 41.176 | 23.614 | 41.971 | 47.851 |
  ```
  maybe we need train with mixup as well.

  updated r2-50-fpn result, add SPP module:

  ```
  |   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
  |:------:|:------:|:------:|:------:|:------:|:------:|
  | 37.460 | 60.874 | 39.977 | 23.974 | 39.842 | 48.025 |
  ```


- 2021.08.26:

  A larger version of r2-50 YOLOv7　(double channel in FPN head) get:

  ```
  |   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
  |:------:|:------:|:------:|:------:|:------:|:------:|
  | 38.201 | 61.046 | 41.049 | 22.268 | 41.216 | 50.649 |
  ```
  we need to know upper bound of this head.

  also, am testing Regnetx-400Mf version, with normal FPN head. <- very important.


- 2021.08.24:

  Updated YOLOX trained result:

  ```
  |   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
  |:------:|:------:|:------:|:------:|:------:|:------:|
  | 37.964 | 57.483 | 40.947 | 23.728 | 42.267 | 47.245 |
  ```
  Almost 38, but we need disable normalnize in YOLOX as newly updated. Also applied FP16 enable to train.
  swin-s YOLOv7 result:

  ```
  |   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
  |:------:|:------:|:------:|:------:|:------:|:------:|
  | 36.530 | 58.207 | 39.378 | 17.871 | 42.366 | 51.968 |
  ```
  I think transformer-based need larger iterations.


- 2021.08.19:
  
  More epochs, and now r2-50 YOLOv7 get a better result:

  ```
  |   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
  |:------:|:------:|:------:|:------:|:------:|:------:|
  | 35.245 | 58.542 | 37.056 | 20.579 | 38.780 | 45.712 |
  ```

  And YOLOX get a better result as well (enable L1 loss at last 40k iters):

  ```
  |   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
  |:------:|:------:|:------:|:------:|:------:|:------:|
  | 37.733 | 57.336 | 40.774 | 22.439 | 41.614 | 48.046 |
  ```

  Next, gonna change YOLOv7 arch, add FPN and PAN, also, add dropblocks.
  IoU aware training and SPP.


- 2021.08.17:

  I trained yolov5 again, if i bigger anchor_t -> 4.0, it can work a little:

  ```
  |   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
  |:------:|:------:|:------:|:------:|:------:|:------:|
  | 35.814 | 70.814 | 33.020 | 10.189 | 32.159 | 43.572 |
  ```
  but small result is poor, i also used YOLOv5 official coco's anchor settings.
  so, does it work or not work? Hard to say.
  

- 2021.08.16:
  
  Finally, get result YOLOX trained AP:

  ```
  [08/14 06:28:23 d2.evaluation.coco_evaluation]: Evaluation results for bbox: 
  |   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
  |:------:|:------:|:------:|:------:|:------:|:------:|
  | 36.572 | 56.028 | 39.497 | 22.921 | 40.583 | 46.090 |

  ```
  it not using stop augmentation trick and enable l1 loss at last 15 epochs. the overall iterations can be longger than 120000, etc. 180000, lr 0.02 -> 0.03

  and r2-50 model:

  ```
  [08/14 08:13:06 d2.evaluation.coco_evaluation]: Evaluation results for bbox: 
  |   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
  |:------:|:------:|:------:|:------:|:------:|:------:|
  | 34.950 | 58.071 | 37.104 | 20.319 | 37.877 | 45.645 |
  ```

  things needed to be added to YOLOv7:
  - Dropblock;
  - SPP;
  - fpn and pan;
  - enable l1 loss at last iterations;
  - decouple head;
  - IoU aware training.


- 2021.08.12:

  Now, we can reveal YOLOX eval result, but we have to train it, we forced using BGR as input order,
  rather than RGB. since we don't want swap channel when opencv read the image, directly using BGR order in opencv.

  ```
  [08/12 01:08:32 d2.evaluation.coco_evaluation]: Evaluation results for bbox: 
  |   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
  |:------:|:------:|:------:|:------:|:------:|:------:|
  | 25.162 | 43.746 | 25.611 | 14.494 | 29.015 | 31.481 |

  ```

  I still can not get a good mAP on r2-50.
  But I found now, using YOLOX can achieve 28.9 mAP. (By changing the dataset and some params),
  YOLOX can get a resonable AP:

  ```
  |   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
  |:------:|:------:|:------:|:------:|:------:|:------:|
  | 35.181 | 54.611 | 38.210 | 21.914 | 39.098 | 44.240 |

  ```
  Now get a good 35 mAP for YOLOX.
  Once achieve to 38 or 37 than it can be reprecated.

  Also, r2-50 get a good mAP now:

  ```
  |   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
  |:------:|:------:|:------:|:------:|:------:|:------:|
  | 30.731 | 54.494 | 31.426 | 16.675 | 33.858 | 39.642 |

  ```
  

- 2021.08.11:

  I try to reveal eval result of YOLOX, I using exactly same weights from YOLOX, first
  I found the AP is 31, far less than 39 claimed in YOLOX. Finally found it was because of am using BGR format
  by default, but YOLOX using RGB format.

  After I change, the AP seems normal a little bit:

  ```
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.351
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.535
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.384
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.236
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.406
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.406
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.283
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.434
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.447
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.296
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.497
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.524
  [08/11 16:57:32 d2.evaluation.coco_evaluation]: Evaluation results for bbox: 
  |   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
  |:------:|:------:|:------:|:------:|:------:|:------:|
  | 35.138 | 53.534 | 38.365 | 23.565 | 40.614 | 40.553 |
  ```

  I changed the padding value from 0 to 114/255, now I can get a very close AP using YOLOX pretrained model:

  ```
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.386
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.589
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.417
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.239
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.442
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.472
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.316
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.515
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.551
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.380
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.607
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.652
  [08/11 17:44:11 d2.evaluation.coco_evaluation]: Evaluation results for bbox: 
  |   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
  |:------:|:------:|:------:|:------:|:------:|:------:|
  | 38.619 | 58.881 | 41.690 | 23.898 | 44.153 | 47.201 |
  ```
  Now get a better one:

  ```
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.389
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.589
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.421
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.237
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.441
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.486
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.318
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.526
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.575
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.403
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.635
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.684
  [08/11 18:07:58 d2.evaluation.coco_evaluation]: Evaluation results for bbox: 
  |   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
  |:------:|:------:|:------:|:------:|:------:|:------:|
  | 38.910 | 58.889 | 42.128 | 23.701 | 44.142 | 48.574 |

  ```

  However, still can not achieve YOLOX train, best now:

  ```
  [08/11 15:41:59 d2.evaluation.coco_evaluation]: Evaluation results for bbox: 
  |   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
  |:------:|:------:|:------:|:------:|:------:|:------:|
  | 24.360 | 40.181 | 25.170 | 13.731 | 26.986 | 29.080 |
  ```


- 2021.08.07:

  I found using new masic and fixed size might help a little bit result, but trained on VisDrone still not good:

  iter 80000:
  ```
  |   AP   |  AP50  |  AP75  |  APs  |  APm   |  APl   |
  |:------:|:------:|:------:|:-----:|:------:|:------:|
  | 15.049 | 33.972 | 11.666 | 8.957 | 21.881 | 21.588 |  
  ```
  why the AP so low?

  Next experiment, I need reveal YOLOX result, since everything is totally same.

  - I added RandomeResizeShortest to yolov7 datamapper, so that it can handle multi-scale inputs;
  - Train 1w to see how visdrone will effected, also we enables l1 loss in training by default;

  I got mAP 21, mAP50 40 on visdrone. Not so bad. At least it seems low learning can avoid to local optimal.

  got mAP 23, mAP50 41.4 using YOLOXs.
  it seems still a gap between sota methods:

  ```
  [08/11 09:09:54 d2.evaluation.coco_evaluation]: Evaluation results for bbox:
  |   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
  |:------:|:------:|:------:|:------:|:------:|:------:|
  | 23.423 | 42.075 | 22.554 | 15.606 | 33.356 | 34.577 |
  ```

- 2021.08.06:

  train YOLOX but get bad result:

  iter 60000:

  ```
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.240
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.410
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.250
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.133
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.272
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.289
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.227
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.369
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.392
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.233
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.430
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.470
  [08/06 10:52:53 d2.evaluation.coco_evaluation]: Evaluation results for bbox: 
  |   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
  |:------:|:------:|:------:|:------:|:------:|:------:|
  | 24.039 | 41.046 | 24.978 | 13.258 | 27.167 | 28.879 |
  ```
  How does it possible? Why all models can not get a reasonable AP?
  I suspect it was because of bad data augmentation introduce from YOLOF, I copied mosiac aug from YOLOX and try again.
  Now, new mosiac looks:
  1. disabled resize, using directly input;
  2. new mosiac can specifc input_size, and make it merge to real data to train;
  3. new mosiac reduced zero width boxes.


- 2021.08.04:

  Exp on normal yolov4 loss on voc:

  - Seems loss not drop, why? Same config as coco, but coco at least can drop;

- 2021.08.03:

  Important notes:

  - Using divide to num_fg can get a lower conf loss, training can be stable;
  - obj_mask must multiply conf loss, otherwise you will miss a lot fg objects.
  
  Original yolov3 loss actually can work. we need stick with it. Push it's limitions.
  On the other hand, we need make YOLOv5 also able training, so that we can fuse YOLOv5 target tricks easily.
  Waiting for YOLOX's training result.

- 2021.08.01:

  I tried train on some large dataset such as VOC, currently tested r2-50 YOLOv7 arch we got:

  ```
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.352
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.809
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.247
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.340
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.395
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.343
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.275
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.436
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.448
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.384
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.461
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.442
  [08/01 12:04:43 d2.evaluation.coco_evaluation]: Evaluation results for bbox:
  |   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
  |:------:|:------:|:------:|:------:|:------:|:------:|
  | 35.193 | 80.946 | 24.743 | 33.998 | 39.462 | 34.337 |
  ```
  I think AP50 slight lower than standard YOLOv3, but seems the whole arch has no problem.
  Next try test x-s-pafpn result with loss enhancement. Make sure it can perform well.

  Method to achieve a higher AP:
  - Compact head design;
  - SPP + PAN should perform better than r2-50, exp it;
  - label smoothing + MixUp;

  - Train YOLOX, is the augmentation useful or not?

- 2021.07.26:

  2 things need to do for now:

  1). xs_pafpn seems have a stable performance, reproduced it with batchsize 128;
  2). Reproduce coco result with cocomini, with batchsize 128 train;
  3). Get a reasonable AP on VOC or coco.

  Also, please fix tl val problem.

  I found coco hard to make it converge, or AP boost easily.... Hard to make it fully trained.
  I should also train a res2net18 lite model, try what will happen with it.

- 2021.07.25:

  We got a new record!! res2net50_v1d achieved a better result:

  ```
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.649
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.980
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.796
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.449
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.654
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.659
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.477
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.690
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.704
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.497
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.711
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.713
  [07/25 09:46:38 d2.evaluation.coco_evaluation]: Evaluation results for bbox:
  |   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
  |:------:|:------:|:------:|:------:|:------:|:------:|
  | 64.912 | 97.988 | 79.599 | 44.884 | 65.412 | 65.926 |
  [07/25 09:46:38 d2.evaluation.coco_evaluation]: Per-category bbox AP:
  | category   | AP     | category   | AP     |
  |:-----------|:-------|:-----------|:-------|
  | face       | 64.081 | face_mask  | 65.743 |
  [07/25 09:46:38 d2.engine.defaults]: Evaluation results for facemask_val in csv format:
  [07/25 09:46:38 d2.evaluation.testing]: copypaste: Task: bbox
  [07/25 09:46:38 d2.evaluation.testing]: copypaste: AP,AP50,AP75,APs,APm,APl
  [07/25 09:46:38 d2.evaluation.testing]: copypaste: 64.9119,97.9876,79.5991,44.8837,65.4117,65.9264
  ```

  mAP 64! above res50 a lot!

- 2021.07.22:
  
  Why resnet doesn't work at all??? Even CSP-Darknet not works very well.
  Same config with facemask_cspdarknet53, I train another r50 backbone. 

  r50 v7 loss  get a better AP:

  ```
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.599
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.963
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.688
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.386
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.608
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.609
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.449
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.646
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.658
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.420
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.667
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.667
  [07/23 17:55:58 d2.evaluation.coco_evaluation]: Evaluation results for bbox:
  |   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
  |:------:|:------:|:------:|:------:|:------:|:------:|
  | 59.950 | 96.304 | 68.820 | 38.601 | 60.775 | 60.892 |
  [07/23 17:55:58 d2.evaluation.coco_evaluation]: Per-category bbox AP:
  | category   | AP     | category   | AP     |
  |:-----------|:-------|:-----------|:-------|
  | face       | 58.198 | face_mask  | 61.702 |
  [07/23 17:55:58 d2.engine.defaults]: Evaluation results for facemask_val in csv format:
  [07/23 17:55:58 d2.evaluation.testing]: copypaste: Task: bbox
  [07/23 17:55:58 d2.evaluation.testing]: copypaste: AP,AP50,AP75,APs,APm,APl
  [07/23 17:55:58 d2.evaluation.testing]: copypaste: 59.9499,96.3041,68.8203,38.6014,60.7751,60.8919
  [07/23 17:55:58 d2.utils.events]:  eta: 23:19:17  iter: 119999  total_loss: 2.492  loss_box: 0.3664
  ```
  mAP 59.95! I found this AP might because of resnet50 output channels are: 512, 1024, 2048,
  while dakrnet are 256, 512, 1024  

  I try reduce channel to 256 for resnet:
  input 512: 
  ```
  res3 torch.Size([1, 256, 64, 64])
  res4 torch.Size([1, 512, 32, 32])
  res5 torch.Size([1, 1024, 16, 16])
  ```
  same as darknet.

  **I found res50 can not be train without pretrained model. Also the channel output can not be changed**.


- 2021.07.21:
  
  I found the way I using ciou has bug, it can not benifit model performance at all.
  Debugging on it...

  Got the first result with ciou correct:

  ```
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.494
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.909
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.476
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.269
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.523
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.498
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.386
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.567
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.578
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.308
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.598
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.584
  [07/22 09:07:35 d2.evaluation.coco_evaluation]: Evaluation results for bbox:
  |   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
  |:------:|:------:|:------:|:------:|:------:|:------:|
  | 49.372 | 90.876 | 47.579 | 26.898 | 52.327 | 49.841 |
  [07/22 09:07:35 d2.evaluation.coco_evaluation]: Per-category bbox AP:
  | category   | AP     | category   | AP     |
  |:-----------|:-------|:-----------|:-------|
  | face       | 51.021 | face_mask  | 47.723 |
  ```
  mAP 49.3! 

- 2021.07.19:
  
  Going test the this 2 trick can work or not:
  1. ciou loss;
  2. mosiac augmentation;
  3. larger lr better results?
  4. r50-fpn output channel set to 1024 gain improvements?

  lr too large make coco hard to converge, actually it is too big for the first serveral experiments. now try:

  1. cspdarknet: be better accuracy;
  2. Does ciou work?
  3. Try YOLOX head design, darknet + pafpn head design;
   


- 2021.07.07: coco-r50-pan seems work now. with all augmentation open. but trafficlight not work seems center point were shifted. (Problem solved)
  a. tl center shifted, is it anchor reason or something?
  b. mosiac augmentation actually works;
  c. 

  Above problem solved mainly by 2 reasons:
  1. the `demo.py` preprocess step not correctly aligned;
  2. the IGNOR_THRESHOLD set too low, this will effect training badly.