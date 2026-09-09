"""P2 全通道验证用 tiny CNN：生成 ONNX + 参考输入输出 bin。

固定随机种子；输入/期望输出落盘为 raw float32（NCHW），
供真机 App 侧加载 OM 推理后逐值对拍（input.bin / expected.bin）。
算子集取自 Model Zoo 经典 CNN（Conv/Relu/MaxPool/GlobalAveragePool/Reshape/Gemm），
用于验证 OMG kirinx90 转换与 NNRt HIAI_F 加载推理全通道，不含任何 LLM 专属算子。
"""
import numpy as np
import onnx
from onnx import helper, TensorProto

rng = np.random.RandomState(0)

w1 = (rng.randn(8, 3, 3, 3) * 0.05).astype(np.float32)    # conv1 [8,3,3,3]
b1 = (rng.randn(8) * 0.01).astype(np.float32)
w2 = (rng.randn(16, 8, 3, 3) * 0.05).astype(np.float32)   # conv2 [16,8,3,3]
b2 = (rng.randn(16) * 0.01).astype(np.float32)
w3 = (rng.randn(10, 16) * 0.05).astype(np.float32)        # gemm B [10,16] transB=1
b3 = (rng.randn(10) * 0.01).astype(np.float32)

inits = [
    helper.make_tensor("conv1_w", TensorProto.FLOAT, w1.shape, w1.tobytes(), raw=True),
    helper.make_tensor("conv1_b", TensorProto.FLOAT, b1.shape, b1.tobytes(), raw=True),
    helper.make_tensor("conv2_w", TensorProto.FLOAT, w2.shape, w2.tobytes(), raw=True),
    helper.make_tensor("conv2_b", TensorProto.FLOAT, b2.shape, b2.tobytes(), raw=True),
    helper.make_tensor("fc_w", TensorProto.FLOAT, w3.shape, w3.tobytes(), raw=True),
    helper.make_tensor("fc_b", TensorProto.FLOAT, b3.shape, b3.tobytes(), raw=True),
    onnx.numpy_helper.from_array(np.array([1, 16], dtype=np.int64), "shape_16"),
]

nodes = [
    helper.make_node("Conv", ["input", "conv1_w", "conv1_b"], ["c1"],
                     kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
    helper.make_node("Relu", ["c1"], ["r1"]),
    helper.make_node("MaxPool", ["r1"], ["p1"], kernel_shape=[2, 2], strides=[2, 2]),
    helper.make_node("Conv", ["p1", "conv2_w", "conv2_b"], ["c2"],
                     kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
    helper.make_node("Relu", ["c2"], ["r2"]),
    helper.make_node("GlobalAveragePool", ["r2"], ["gap"]),      # [1,16,1,1]
    helper.make_node("Reshape", ["gap", "shape_16"], ["flat"]),  # [1,16]
    helper.make_node("Gemm", ["flat", "fc_w", "fc_b"], ["output"],
                     alpha=1.0, beta=1.0, transB=1),
]

graph = helper.make_graph(
    nodes, "tiny_cnn",
    [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 32, 32])],
    [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10])],
    inits,
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 8  # omg 解析器偏旧，压低 IR 版本求兼容
onnx.checker.check_model(model)
onnx.save(model, "cann/tiny_cnn.onnx")

# 参考输入输出：纯 python 参考实现求值，runner 无需 onnxruntime
from onnx.reference import ReferenceEvaluator

x = np.full((1, 3, 32, 32), 0.1, dtype=np.float32)
y = ReferenceEvaluator(model).run(None, {"input": x})[0].astype(np.float32)
x.tofile("cann/input.bin")
y.tofile("cann/expected.bin")
print("input :", x.flatten()[:4], "...")
print("output:", np.round(y.flatten(), 4))
print("saved: cann/tiny_cnn.onnx, cann/input.bin, cann/expected.bin")
