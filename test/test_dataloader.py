import os
from detectron2.engine import default_argument_parser
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets.coco import register_coco_instances
from detectron2.data import build_detection_train_loader
from yolov7.data.dataset_mapper import MyDatasetMapper
from detectron2.structures.masks import BitMasks
import torch
from alfred.vis.image.det import visualize_det_cv2_part
from alfred.vis.image.mask import vis_bitmasks_with_classes
from skimage.transform import resize
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
import sys

def visualize(save_path,**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    
    plt.tight_layout()
    
    if save_path:
       plt.savefig(save_path,  bbox_inches='tight')
    plt.show()


def vis_res_fast(res, img, class_names, colors, thresh):
    ins = res["instances"]
    bboxes = None
    if ins.has("gt_boxes"):
        bboxes = ins.gt_boxes.tensor.cpu().numpy()
    scores = None#ins.scores.cpu().numpy()
    clss = ins.gt_classes.cpu().numpy()
    if ins.has("gt_masks"):
        bit_masks = ins.gt_masks
        if isinstance(bit_masks, BitMasks):
            bit_masks = bit_masks.tensor.cpu().numpy()
        
        if isinstance(bit_masks, torch.Tensor):
            bit_masks = bit_masks.cpu().numpy()
        new_masks  = []
        width = img.shape[0]
        height = img.shape[1]
        for index in range(bit_masks.shape[0]):
            mask = bit_masks[index,::]
            mask = resize(mask,(width, height))
            new_masks.append(mask)
        bit_masks = np.array(new_masks)
        # img = vis_bitmasks_with_classes(img, clss, bit_masks)
        # img = vis_bitmasks_with_classes(img, clss, bit_masks, force_colors=colors, mask_border_color=(255, 255, 255), thickness=2)
        img = vis_bitmasks_with_classes(
            img, clss, bit_masks,class_names=class_names, force_colors=None, draw_contours=True, alpha=0.8
        )

    if ins.has("pred_masks"):
        bit_masks = ins.pred_masks
        if isinstance(bit_masks, BitMasks):
            bit_masks = bit_masks.tensor.cpu().numpy()
        img = vis_bitmasks_with_classes(
            img,
            clss,
            bit_masks,
            class_names=class_names,
            force_colors=None,
            draw_contours=True,
            alpha=0.6,
            thickness=2,
        )
    thickness = 1 if ins.has("gt_masks") else 2
    font_scale = 0.3 if ins.has("gt_masks") else 0.4
    if bboxes is not None:
        img = visualize_det_cv2_part(
            img,
            scores,
            clss,
            bboxes,
            class_names=class_names,
            force_color=colors,
            line_thickness=thickness,
            font_scale=font_scale,
            thresh=thresh,
        )
    return img

if __name__ == '__main__':
    # need modify here if using other models
    from train_inseg import setup
    
    parser = default_argument_parser()
    parser.add_argument("--count", default=30, type=int, help="generate image count")
    parser.add_argument("--anno-file", default="", metavar="FILE", help="annotation file path")
    parser.add_argument("--img-path", default="", help="image file path")
    parser.add_argument("--output-path", default="", help="augement and label result image save path")
    args = parser.parse_args()
    if len(sys.argv) < 2:
        args.output_path = '../tmp'
        args.config_file ='../configs/coco-instance/yolomask.yaml'
        args.anno_file = "../datasets/fruit_segmentation.v4i.coco-segmentation/train/_annotations.coco.json"
        args.img_path = "../datasets/fruit_segmentation.v4i.coco-segmentation/train"
        
    else:
        pass
        
    # cfg is init from .\yolov7_d2\yolov7\config.py
    args.num_gpus = 1
    
    cfg = setup(args)
    # modeify here if you want change cfg config
    cfg.defrost()
    # data aug implement init at .\data\detection_utils.py
    cfg.DATASETS.CLASS_NAMES = ['fruits','apple','lemon','orange','pear','strawberry']
    #cfg.INPUT.SHIFT.ENABLED = True
    #cfg.INPUT.SHIFT.SHIFT_PIXELS = 12
    #cfg.INPUT.RANDOM_FLIP_HORIZONTAL.PROB = 0.1
    #cfg.INPUT.RANDOM_FLIP_HORIZONTAL.ENABLED = True
    #cfg.INPUT.RANDOM_FLIP_VERTICAL.PROB = 0.9
    #cfg.INPUT.RANDOM_FLIP_VERTICAL.ENABLED = True
    #cfg.INPUT.MAX_SIZE_TRAIN = 320
    #cfg.INPUT.MIN_SIZE_TRAIN  = (320,)
    #cfg.INPUT.MAX_SIZE_TEST = 320
    #cfg.INPUT.MIN_SIZE_TEST = 320
    cfg.freeze()
    
    MetadataCatalog.clear()
    DatasetCatalog.clear()
    register_coco_instances(cfg.DATASETS.TRAIN[0], {}, args.anno_file, args.img_path)
    
    custom_mapper = MyDatasetMapper(cfg, True)
    #results = build_detection_train_loader(cfg, mapper=custom_mapper,aspect_ratio_grouping = False)
    results = build_detection_train_loader(cfg, mapper=custom_mapper)
    class_names = cfg.DATASETS.CLASS_NAMES
    colors = [
        [random.randint(0, 255) for _ in range(3)] # random R,G,B value
        for _ in range(len(class_names)) # class count
    ]
    
    count = 0
    obj = None
    print('*** display dataset , press ctrl+c to stop ***')
    for item in results:
        
        try:
            
            count+=1
            if count>args.count:
                break
            #print('count=',count)
            obj = item
            file_name = item[0]['file_name']
            img = cv2.imread(file_name)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            #cv2_imshow(img)
            
            image = item[0]['image'].cpu().numpy()
            image = image.transpose(1, 2, 0)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            prev, name = os.path.split(item[0]['file_name'])
            out_file_name = ''
            if os.path.isdir(args.output_path):
                out_file_name = os.path.join(args.output_path,name)
            aug_img = image.copy()
            result = vis_res_fast(item[0],image,class_names,colors,0)

            visualize(out_file_name,original_img = img,aug_img=aug_img,aug_label_img = result)
            pass
        
        except KeyboardInterrupt:
            print('stop')
            break