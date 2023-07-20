#!/bin/bash -x

hhb -D --model-file ../../model/mobilenetv2-12.onnx --calibrate-dataset persian_cat.jpg --data-scale 0.017 --data-mean "124 117 104" --board th1520 --input-name "input" --output-name "output" --input-shape "1 3 224 224" --postprocess save_and_top5 --quantization-scheme="int16_sym" --input-memory-type 1
