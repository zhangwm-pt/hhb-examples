#!/bin/bash -x

hhb -f ../../model/mobilenetv2-12.onnx --data-scale 0.017 --data-mean "104 117 124" --board c920 -D --postprocess save_and_top5  --model-save save_and_run -in "input" -on "output" -is "1 3 224 224" -o jit # -v -v -v 
hhb -f ../../model/mobilenetv2-12.onnx --data-scale 0.017 --data-mean "104 117 124" --board c920 -D --postprocess save_and_top5  --model-save save_and_run -in "input" -on "output" -is "1 3 224 224" -o runtime --link-lib shl_th1520 # -v -v -v 
cd jit
qemu-riscv64 -cpu c920 hhb_jit hhb.bm
cp shl.hhb.bm ../runtime
