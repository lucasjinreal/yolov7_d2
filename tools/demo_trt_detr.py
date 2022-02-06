
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


import os
import sys
import time
import cv2
from PIL import Image
import argparse

import pycuda.driver as cuda
import pycuda.autoinit
# import cupy as cp
import numpy as np
import tensorrt as trt
import glob

# from trt_util.common import allocate_buffers, do_inference_v2, build_engine_onnx
from alfred.deploy.tensorrt.common import allocate_buffers, do_inference_v2, build_engine_onnx, allocate_buffers_v2
# from trt_util.process_img import preprocess_np, preprocess_torch_v1
# from trt_util.plot_box import plot_box, CLASSES
from alfred.vis.image.det import visualize_det_cv2_part, visualize_det_cv2_fancy
from torch import onnx


TRT_LOGGER = trt.Logger(trt.Logger.INFO)
means = [123.675, 116.280, 103.530]
stds = [58.395, 57.120, 57.375]

def preprocess_np_no_normalize(img_path):
    im = cv2.imread(img_path)
    print(img_path)
    print(im.shape)
    # img = transform(im).unsqueeze(0)
    # a = cv2.resize(im, (960, 768))
    a = cv2.resize(im, (1960, 1080))
    a = a.astype(np.float32)
    # a -= means
    # a /= stds
    a = np.transpose(a, (2, 0, 1)).astype(np.float32)
    a = np.expand_dims(a, axis=0)
    return a, im


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



def engine_infer(engine, context, inputs, outputs, bindings, stream, test_image):

    # image_input, img_raw, _ = preprocess_np(test_image)
    image_input, img_raw = preprocess_np_no_normalize((test_image))
    print(image_input.shape)
    # device to host to device,在性能对比时将替换该方式
    inputs[0].host = image_input.astype(np.float32).ravel()

    start = time.time()
    res = do_inference_v2(context, bindings=bindings, inputs=inputs,
                          outputs=outputs, stream=stream, input_tensor=image_input)
    print(f"推断耗时：{time.time()-start}s")
    res = res[0]

    # print(scores)
    # print(boxs)

    output_shapes = [(1, 100, 4 + 2)]
    # scores = scores.reshape(output_shapes[0])
    # boxs = boxs.reshape(output_shapes[1])
    res = res.reshape(output_shapes[0])
    # print(res)
    return res, img_raw


def vis_res_fast(res, img):
    # res = res[0].cpu().numpy()
    print(res.shape)
    res = res[0]
    scores = res[:, -2]
    clss = res[:, -1]
    bboxes = res[:, :4]

    indices = scores > 0.6
    bboxes = bboxes[indices]
    scores = scores[indices]
    clss = clss[indices]

    img = visualize_det_cv2_part(
        img, scores, clss, bboxes, is_show=True)
    # img = cv2.addWeighted(img, 0.9, m, 0.6, 0.9)
    return img


def main(onnx_model_file, image_dir, fp16=False, int8=False, batch_size=1, dynamic=False):
    test_images = glob.glob(os.path.join(image_dir, '*.jpg'))
    sorted(test_images)

    if onnx_model_file.endswith('onnx'):
        engine_file = onnx_model_file + '.trt'
    else:
        engine_file = onnx_model_file

    if int8:
        # only load the plan engine file
        if not os.path.exists(engine_file):
            raise "[Error] INT8 Mode must given the correct engine plan file. Please Check!!!"
        with open(engine_file, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        with engine.create_execution_context() as context:
            inputs, outputs, bindings, stream = allocate_buffers(engine)
            # print(dir(context))

            if dynamic:
                context.active_optimization_profile = 0  # 增加部分
                origin_inputshape = context.get_binding_shape(0)
                if origin_inputshape[0] == -1:
                    origin_inputshape[0] = batch_size
                    context.set_binding_shape(0, (origin_inputshape))
            print(
                f"[INFO] INT8 mode.Dynamic:{dynamic}. Deserialize from: {engine_file}.")

            for test_image in test_images:

                scores, boxs, img_raw = engine_infer(
                    engine, context, inputs, outputs, bindings, stream, os.path.join(image_dir, test_image))

                print(
                    f"[INFO] trt inference done. save result in : ./trt_infer_res/in8/{test_image}")
                if not os.path.exists("./results/in8"):
                    os.makedirs("./results/in8")
                plot_box(img_raw, scores, boxs, prob_threshold=0.7,
                         save_fig=os.path.join('./trt_infer_res/in8', test_image))

    else:
        # Build a TensorRT engine.
        with build_engine_onnx(onnx_model_file, engine_file, FP16=fp16, verbose=False,
                               dynamic_input=dynamic, chw_shape=[3, 768, 960]) as engine:
            # inputs, outputs, bindings, stream = allocate_buffers_v2(engine)
            inputs, outputs, bindings, stream = allocate_buffers(engine)
            # Contexts are used to perform inference.
            with engine.create_execution_context() as context:
                print(engine.get_binding_shape(0))
                print(engine.get_binding_shape(1))
                # Load a normalized test case into the host input page-locked buffer.
                if dynamic:
                    context.active_optimization_profile = 0  # 增加部分
                    origin_inputshape = context.get_binding_shape(0)
                    if origin_inputshape[0] == -1:
                        origin_inputshape[0] = batch_size
                        context.set_binding_shape(0, (origin_inputshape))

                print(
                    f"[INFO] FP16 mode is: {fp16},Dynamic:{dynamic} Deserialize from: {engine_file}.")

                for test_image in test_images:
                    res, img_raw = engine_infer(
                        engine, context, inputs, outputs, bindings, stream, test_image)

                    if fp16:
                        save_dir = "./results/fp16"
                    else:
                        save_dir = "./results/fp32"

                    print(
                        f"[INFO] trt inference done. save result in : {save_dir}/{test_image}")
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    # plot_box(img_raw, scores, boxs, prob_threshold=0.7,
                    #          save_fig=os.path.join(save_dir, test_image))
                    print(res)
                    res = detr_postprocess(res, img_raw)
                    vis_res_fast(res, img_raw)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Inference by TensorRT in FP32 ,FP16 Mode or INT8 Mode.')
    parser.add_argument('--model_dir', type=str,
                        default='./detr_sim.onnx', help='ONNX Model Path')
    parser.add_argument('--image_dir', type=str,
                        default="./images", help='Test Image Dir')

    parser.add_argument('--fp16', action="store_true",
                        help='Open FP16 Mode or Not, if True You Should Load FP16 Engine File')
    parser.add_argument('--int8', action="store_true",
                        help='Open INT8 Mode or Not, if True You Should Load INT8 Engine File')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size, static=1')
    parser.add_argument('--dynamic', action="store_true",
                        help='Dynamic Shape or Not when inference in trt')

    args = parser.parse_args()

    main(args.model_dir, args.image_dir,
         args.fp16, args.int8, args.batch_size, args.dynamic)
