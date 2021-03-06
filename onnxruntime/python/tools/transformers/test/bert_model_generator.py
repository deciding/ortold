# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import onnx
import math
from typing import List
from packaging import version
from onnx import helper, TensorProto


def float_tensor(name: str, shape: List[int], random=False):
    low = 0.0
    high = 1.0
    total_elements = 1
    for x in shape:
        total_elements *= x
    weights = [random.uniform(low, high) for _ in range(total_elements)] if random else [1.0] * total_elements
    return helper.make_tensor(name, TensorProto.FLOAT, shape, weights)


def create_bert_attention(input_hidden_size=16, pruned_num_heads=2, pruned_head_size=4, use_float_mask=False):
    # unsqueeze in opset version 13 has two inputs (axis is moved from attribute to input).
    has_unsqueeze_two_inputs = (version.parse(onnx.__version__) >= version.parse('1.8.0'))

    # nodes in attention subgraph
    nodes = [
        helper.make_node("Add", ["input_1", "input_2"], ["layernorm_input"], "add_layernorm"),
        helper.make_node("LayerNormalization", ["layernorm_input", "layer_norm_weight", "layer_norm_bias"],
                         ["layernorm_out"],
                         "layernorm",
                         axis=-1,
                         epsion=0.000009999999747378752),

        # q nodes
        helper.make_node("MatMul", ["layernorm_out", "matmul_q_weight"], ["matmul_q_out"], "matmul_q"),
        helper.make_node("Add", ["matmul_q_out", "add_q_weight"], ["add_q_out"], "add_q"),
        helper.make_node("Reshape", ["add_q_out", "reshape_weight_1"], ["reshape_q_out"], "reshape_q"),
        helper.make_node("Transpose", ["reshape_q_out"], ["transpose_q_out"], "transpose_q", perm=[0, 2, 1, 3]),

        # k nodes
        helper.make_node("MatMul", ["layernorm_out", "matmul_k_weight"], ["matmul_k_out"], "matmul_k"),
        helper.make_node("Add", ["matmul_k_out", "add_k_weight"], ["add_k_out"], "add_k"),
        helper.make_node("Reshape", ["add_k_out", "reshape_weight_1"], ["reshape_k_out"], "reshape_k"),
        helper.make_node("Transpose", ["reshape_k_out"], ["transpose_k_out"], "transpose_k", perm=[0, 2, 3, 1]),

        # mask nodes
        helper.make_node("Unsqueeze", ["input_mask", "axes_1"], ["unsqueeze0_out"], "unsqueeze0") if has_unsqueeze_two_inputs \
            else helper.make_node("Unsqueeze", ["input_mask"], ["unsqueeze0_out"], "unsqueeze0", axes=[1]),
        helper.make_node("Unsqueeze", ["unsqueeze0_out", "axes_2"], ["unsqueeze1_out"], "unsqueeze1") if has_unsqueeze_two_inputs \
            else helper.make_node("Unsqueeze", ["unsqueeze0_out"], ["unsqueeze1_out"], "unsqueeze1", axes=[2]),

        # when attention_mask is float type, no need to cast
        helper.make_node("Cast", ["unsqueeze1_out"], ["cast_out"], "cast", to=1) if not use_float_mask else None,
        helper.make_node("Sub", ["sub_weight", "unsqueeze1_out" if use_float_mask else "cast_out"], ["sub_out"], "sub"),
        helper.make_node("Mul", ["sub_out", "mul_weight"], ["mul_mask_out"], "mul_mask"),

        # qk nodes
        helper.make_node("MatMul", ["transpose_q_out", "transpose_k_out"], ["matmul_qk_out"], "matmul_qk"),
        helper.make_node("Div", ["matmul_qk_out", "div_weight"], ["div_qk_out"], "div_qk"),
        helper.make_node("Add", ["div_qk_out", "mul_mask_out"], ["add_qk_out"], "add_qk"),
        helper.make_node("Softmax", ["add_qk_out"], ["softmax_qk_out"], "softmax_qk", axis=3),

        # v nodes
        helper.make_node("MatMul", ["layernorm_out", "matmul_v_weight"], ["matmul_v_out"], "matmul_v"),
        helper.make_node("Add", ["matmul_v_out", "add_v_weight"], ["add_v_out"], "add_v"),
        helper.make_node("Reshape", ["add_v_out", "reshape_weight_1"], ["reshape_v_out"], "reshape_v"),
        helper.make_node("Transpose", ["reshape_v_out"], ["transpose_v_out"], "transpose_v", perm=[0, 2, 1, 3]),

        # qkv nodes
        helper.make_node("MatMul", ["softmax_qk_out", "transpose_v_out"], ["matmul_qkv_1_out"], "matmul_qkv_1"),
        helper.make_node("Transpose", ["matmul_qkv_1_out"], ["transpose_qkv_out"], "transpose_qkv", perm=[0, 2, 1, 3]),
        helper.make_node("Reshape", ["transpose_qkv_out", "reshape_weight_2"], ["reshape_qkv_out"], "reshape_qkv"),
        helper.make_node("MatMul", ["reshape_qkv_out", "matmul_qkv_weight"], ["matmul_qkv_2_out"], "matmul_qkv_2"),
        helper.make_node("Add", ["matmul_qkv_2_out", "add_qkv_weight"], ["add_qkv_out"], "add_qkv"),
        helper.make_node("Add", ["add_qkv_out", "layernorm_out"], ["skip_output"], "add_skip"),
        helper.make_node("LayerNormalization", ["skip_output", "layer_norm_weight", "layer_norm_bias"], ["output"],
                         "layernorm2",
                         axis=-1,
                         epsion=0.000009999999747378752),
    ]

    pruned_hidden_size = pruned_num_heads * pruned_head_size
    initializers = [  # initializers
        float_tensor('layer_norm_weight', [input_hidden_size]),
        float_tensor('layer_norm_bias', [input_hidden_size]),
        float_tensor('matmul_q_weight', [input_hidden_size, pruned_hidden_size]),
        float_tensor('matmul_k_weight', [input_hidden_size, pruned_hidden_size]),
        float_tensor('matmul_v_weight', [input_hidden_size, pruned_hidden_size]),
        float_tensor('matmul_qkv_weight', [pruned_hidden_size, input_hidden_size]),
        float_tensor('add_q_weight', [pruned_hidden_size]),
        float_tensor('add_k_weight', [pruned_hidden_size]),
        float_tensor('add_v_weight', [pruned_hidden_size]),
        float_tensor('add_qkv_weight', [input_hidden_size]),
        helper.make_tensor('div_weight', TensorProto.FLOAT, [1], [math.sqrt(pruned_head_size)]),
        helper.make_tensor('sub_weight', TensorProto.FLOAT, [1], [1.0]),
        helper.make_tensor('mul_weight', TensorProto.FLOAT, [1], [-10000]),
        helper.make_tensor('reshape_weight_1', TensorProto.INT64, [4], [0, 0, pruned_num_heads, pruned_head_size]),
        helper.make_tensor('reshape_weight_2', TensorProto.INT64, [3], [0, 0, pruned_hidden_size]),
    ]

    if has_unsqueeze_two_inputs:
        initializers.append(helper.make_tensor('axes_1', TensorProto.INT64, [1], [1]))
        initializers.append(helper.make_tensor('axes_2', TensorProto.INT64, [1], [2]))

    batch_size = 1
    sequence_length = 3
    graph = helper.make_graph(
        [node for node in nodes if node],
        "AttentionFusionPrunedModel",  #name
        [  # inputs
            helper.make_tensor_value_info('input_1', TensorProto.FLOAT,
                                          [batch_size, sequence_length, input_hidden_size]),
            helper.make_tensor_value_info('input_2', TensorProto.FLOAT,
                                          [batch_size, sequence_length, input_hidden_size]),
            helper.make_tensor_value_info('input_mask', TensorProto.FLOAT if use_float_mask else TensorProto.INT64,
                                          [batch_size, sequence_length])
        ],
        [  # outputs
            helper.make_tensor_value_info('output', TensorProto.FLOAT,
                                          [batch_size, sequence_length, input_hidden_size]),
        ],
        initializers)

    model = helper.make_model(graph)
    return model


if __name__ == "__main__":
    model = create_bert_attention()
    onnx.save(model, "pruned_bert_attention.onnx")