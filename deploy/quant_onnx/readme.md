# Quant ONNX

we using tools from ONNXRuntime to directly quantize onnx models and save int8 onnx model.


## Log

- `2022.04.17`: quantize sparseinst and keypoints failed. Seems need all opset=13 to do quant, opset>12 will caused strange result when quantize in onnxruntime;
