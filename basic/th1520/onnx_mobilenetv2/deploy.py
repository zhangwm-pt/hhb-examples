import hhb

hhb.set_debug_level("LOG")
compiler = hhb.Compiler("th1520")
model_file = "../../model/mobilenetv2-12.onnx"
input_name = ["input"]
input_shape = [[1, 3, 224, 224]]
output_name = ["output"]

compiler.import_model(
    model_file,
    input_name=input_name,
    input_shape=input_shape,
    output_name=output_name,
)

compiler.config.preprocess.data_mean.value = [124, 117, 104]
compiler.config.preprocess.data_scale.value = 0.017
compiler.config.postprocess.postprocess.value = "save_and_top5"
compiler.config.codegen.input_memory_type.value = [1]

data_path = "./persian_cat.jpg"
dataset_list = compiler.preprocess(data_path)

compiler.quantize(dataset_list)
compiler.codegen()
compiler.deploy()
