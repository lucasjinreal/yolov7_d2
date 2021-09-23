
#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#                ~~~Medcare AI Lab~~~


import cv2
from PIL import Image
import numpy as np
import os
import time

# onnxruntime requires python 3.5 or above
try:
    import onnxruntime
except ImportError:
    onnxruntime = None

import torch
from torch import nn
import torchvision.transforms as T
from alfred.vis.image.det import visualize_det_cv2_part, visualize_det_cv2_fancy


torch.set_grad_enabled(False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("[INFO] 当前使用{}做推断".format(device))


# 图像数据处理
transform = T.Compose([
    T.Resize((768, 960)),  # PIL.Image.BILINEAR
    T.ToTensor(),
    # T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 将xywh转xyxy


def box_cxcywh_to_xyxy(x):
    x = torch.from_numpy(x)
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

# 将0-1映射到图像


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b.cpu().numpy()
    b = b * np.array([img_w, img_h, img_w, img_h], dtype=np.float32)
    return b

# plot box by opencv


def plot_result(pil_img, prob, boxes, save_name=None, imshow=False, imwrite=False):
    LABEL = ["NA", "Class A", "Class B", "Class C", "Class D", "Class E", "Class F",
             "Class G", "Class H", "Class I", "Class J", "Class K", "Class L", "Class M",
             "Class N", "Class O", "Class P", "Class Q", "Class R", "Class S", "Class T"]
    opencvImage = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    if len(prob) == 0:
        print("[INFO] NO box detect !!! ")
        if imwrite:
            if not os.path.exists("./results/pred_no"):
                os.makedirs("./results/pred_no")
            cv2.imwrite(os.path.join(
                "./results/pred_no", save_name), opencvImage)
        return

    for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes):

        cl = p.argmax()
        if not cl in [6, 7]:
            continue
        label_text = '{}: {}%'.format(LABEL[cl], round(p[cl]*100, 2))

        cv2.rectangle(opencvImage, (int(xmin), int(ymin)),
                      (int(xmax), int(ymax)), (255, 255, 0), 2)
        cv2.putText(opencvImage, label_text, (int(xmin)+10, int(ymin)+30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 0), 2)

    if imshow:
        cv2.imshow('detect', opencvImage)
        cv2.waitKey(0)

    if imwrite:
        if not os.path.exists("./results/pred"):
            os.makedirs('./results/pred')
        cv2.imwrite('./results/pred/{}'.format(save_name), opencvImage)


def detect_onnx(ort_session, img, prob_threshold=0.7):
    # compute onnxruntime output prediction
    # 前处理
    # img = transform(im).unsqueeze(0).cpu().numpy()
    # img = transform(im).cpu().numpy()

    ort_inputs = {"x.1": img}
    start = time.time()
    out = ort_session.run(None, ort_inputs)
    end = time.time()
    print('cost: ', end - start)

    # 后处理 + 也可以加NMS
    return out
    

def vis_res_fast(res, img):
    # res = res[0].cpu().numpy()
    print(res.shape)
    res = res[0]
    scores = res[:, -2]
    clss = res[:, -1]
    bboxes = res[:, :4]

    indices = scores > 0.4
    bboxes = bboxes[indices]
    scores = scores[indices]
    clss = clss[indices]

    img = visualize_det_cv2_part(
        img, scores, clss, bboxes, is_show=True)
    # img = cv2.addWeighted(img, 0.9, m, 0.6, 0.9)
    return img


def detr_postprocess(out_boxes, ori_img):
    """
    normalized xyxy output
    """
    h, w, _ = ori_img.shape
    out_boxes[..., 0] *= w
    out_boxes[..., 1] *= h
    out_boxes[..., 2] *= w
    out_boxes[..., 3] *= h
    return out_boxes

def preprocess_np_no_normalize(img_path):
    im = cv2.imread(img_path)
    print(img_path)
    print(im.shape)
    # img = transform(im).unsqueeze(0)
    a = cv2.resize(im, (960, 768))
    a = np.transpose(a, (2, 0, 1)).astype(np.float32)
    a = np.expand_dims(a, axis=0)
    return a, im


if __name__ == "__main__":

    # onnx_path = "./weights/output.onnx"
    onnx_path = "./weights/detr-r50_sim.onnx"
    d = onnxruntime.get_device()
    print(d)
    ort_session = onnxruntime.InferenceSession(onnx_path)
    files = os.listdir("./images")

    for file in files:
        img_path = os.path.join("./images", file)
        if os.path.isfile(img_path):
            im, ori_img = preprocess_np_no_normalize(img_path)

            out = detect_onnx(ort_session, im)[0]
            print(out.shape)

            # ori_img = np.asarray(im)
            out = detr_postprocess(out, ori_img)
            vis_res_fast(out, ori_img)
            # plot_result(im, scores, boxes,save_name=file,imshow=False, imwrite=True)
            # print("[INFO] {} time: {} done!!!".format(file,None))
