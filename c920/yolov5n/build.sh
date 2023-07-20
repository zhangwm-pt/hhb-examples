#!/bin/bash

#hhb -f yolov5n.onnx -in "images" -on "/model.24/m.0/Conv_output_0;/model.24/m.1/Conv_output_0;/model.24/m.2/Conv_output_0" -is "1 3 384 640" --data-scale-div 255 --board c920 --postprocess save_and_top5 -S -sd kite.jpg --quantization-scheme "float16"

riscv64-unknown-linux-gnu-gcc yolov5n.c -static -o yolov5n_example hhb_out/io.c hhb_out/process.c hhb_out/model.c -Ihhb_out -I/mnt/ssd/zhangwm/git/hhb/install_nn2/c920/include/ -L/mnt/ssd/zhangwm/git/hhb/install_nn2/c920/lib -lshl -lm -I /mnt/ssd/zhangwm/git/csi-nn2/module/decode/install/include/ -L /mnt/ssd/zhangwm/git/csi-nn2/module/decode/install/lib/rv/ -ljpeg -lpng -lz
