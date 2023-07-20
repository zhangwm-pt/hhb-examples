#!/bin/bash -x

hhb -S --model-file ../../model/mobilenetv2-12.onnx  --data-scale 0.017 --data-mean "124 117 104" --board c906 --input-name "input" --output-name "output" --input-shape "1 3 224 224" --postprocess save_and_top5 --simulate-data persian_cat.jpg
