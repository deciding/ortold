need_run=True
if not need_run:
    import os
    import psutil
    os.environ["OMP_NUM_THREADS"] = str(psutil.cpu_count(logical=True))
    os.environ["OMP_WAIT_POLICY"] = 'ACTIVE'

    import onnxruntime
    #import numpy

    output_dir=output_dir = os.path.join("..", "onnx_models")
    export_model_path = os.path.join(output_dir, 'bert-base-cased-squad.onnx')

# Print warning if user uses onnxruntime-gpu instead of onnxruntime package.
    if 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
        print("warning: onnxruntime-gpu is not built with OpenMP. You might try onnxruntime package to test CPU inference.")

    sess_options = onnxruntime.SessionOptions()

# Optional: store the optimized graph and view it using Netron to verify that model is fully optimized.
# Note that this will increase session creation time, so it is for debugging only.
    sess_options.optimized_model_filepath = os.path.join(output_dir, "optimized_model_cpu.onnx")

# For OnnxRuntime 1.2.0, you might need set intra_op_num_threads to 1 to enable OpenMP
#    sess_options.intra_op_num_threads=1
# For OnnxRuntime 1.3.0 or later, it is recommended to use the default setting so you need not set it.

# Specify providers when you use onnxruntime-gpu for CPU inference.
# import pdb;pdb.set_trace()
    session = onnxruntime.InferenceSession(export_model_path, sess_options, providers=['CPUExecutionProvider'])
    exit()



import os
import psutil
import time

# ===DATA=== #
cache_dir = os.path.join("..", "cache_models")
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
predict_file = os.path.join(cache_dir, "dev-v1.1.json")

from transformers import BertTokenizer as tokenizer_class
model_name_or_path = "bert-base-cased"
tokenizer = tokenizer_class.from_pretrained(model_name_or_path, do_lower_case=True, cache_dir=cache_dir)
# load some examples
from transformers.data.processors.squad import SquadV1Processor
processor = SquadV1Processor()
examples = processor.get_dev_examples(None, filename=predict_file)
total_samples=100
max_seq_length=128
max_query_length=64
doc_stride=128

from transformers import squad_convert_examples_to_features
features, dataset = squad_convert_examples_to_features(
            examples=examples[:total_samples], # convert just enough examples for this notebook
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            is_training=False,
            return_dataset='pt'
        )
print('Dataset prepared ...')

# You may change the settings in this cell according to Performance Test Tool result.
os.environ["OMP_NUM_THREADS"] = str(psutil.cpu_count(logical=True))
os.environ["OMP_WAIT_POLICY"] = 'ACTIVE'

import onnxruntime
#import numpy

# ===model=== #
output_dir=output_dir = os.path.join("..", "onnx_models")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
#export_model_path = os.path.join(output_dir, 'bert-base-cased-squad.onnx')
export_model_path = os.path.join(output_dir, 'bert-base-cased-squad_opt_cpu.onnx')

# Print warning if user uses onnxruntime-gpu instead of onnxruntime package.
if 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
    print("warning: onnxruntime-gpu is not built with OpenMP. You might try onnxruntime package to test CPU inference.")

sess_options = onnxruntime.SessionOptions()

# Optional: store the optimized graph and view it using Netron to verify that model is fully optimized.
# Note that this will increase session creation time, so it is for debugging only.
sess_options.optimized_model_filepath = os.path.join(output_dir, "optimized_model_cpu.onnx")
#sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
sess_options.enable_profiling = True
sess_options.log_severity_level = 0
sess_options.log_verbosity_level = 0

# For OnnxRuntime 1.2.0, you might need set intra_op_num_threads to 1 to enable OpenMP
#    sess_options.intra_op_num_threads=1
# For OnnxRuntime 1.3.0 or later, it is recommended to use the default setting so you need not set it.

# Specify providers when you use onnxruntime-gpu for CPU inference.
session = onnxruntime.InferenceSession(export_model_path, sess_options, providers=['CPUExecutionProvider'])

run_options = onnxruntime.RunOptions()
run_options.log_severity_level = 0
run_options.log_verbosity_level = 0

latency = []
for i in range(total_samples):
    data = dataset[i]
    ort_inputs = {
        'input_ids':  data[0].cpu().reshape(1, max_seq_length).numpy(),
        'input_mask': data[1].cpu().reshape(1, max_seq_length).numpy(),
        'segment_ids': data[2].cpu().reshape(1, max_seq_length).numpy()
    }
    start = time.time()
    ort_outputs = session.run(None, ort_inputs, run_options=run_options)
    latency.append(time.time() - start)
print("OnnxRuntime cpu Inference time = {} ms".format(format(sum(latency) * 1000 / len(latency))))
prof_file = session.end_profiling()
print(prof_file)
