#!/bin/bash

#hhb -f mobilenetv2-12.onnx  --calibrate-dataset persian_cat.jpg --data-scale 0.017 --data-mean "124 117 104"  --board c920 -sd persian_cat.jpg -D --postprocess save_and_top5 -in "input" -on "output" -is "1 3 224 224"

riscv64-unknown-linux-gnu-g++ main.cpp -I./prebuilt_opencv/include/opencv4 -L./prebuilt_opencv/lib   -lopencv_imgproc   -lopencv_imgcodecs -L./prebuilt_opencv/lib/opencv4/3rdparty/ -llibjpeg-turbo -llibwebp -llibpng -llibtiff -llibopenjp2    -lopencv_core -ldl  -lpthread -lrt -lzlib -lcsi_cv -latomic -static -o mobilenetv2_example
