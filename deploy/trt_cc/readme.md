run

```
mkdir build
cd build
cmake ..
make -j8
./demo_yolox ../../../weights/coco_yolox_s.trt -i ../../../images/COCO_val2014_000000001869.jpg
```