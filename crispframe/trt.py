"""TensorRT 推理后端：ONNX 导出 → 引擎构建（带缓存）→ 与 torch 共流的零拷贝推理。

- 引擎按 (模型, 输入 shape, 精度) 缓存于 models/engines/，首次构建较慢，之后秒级加载。
- 输入/输出均为 torch CUDA 张量，通过 data_ptr 绑定 + 共享当前 stream，无额外拷贝。
"""
from pathlib import Path

import torch

from .registry import PRETRAINED_DIR

ENGINES_DIR = PRETRAINED_DIR.parent / "engines"

try:
    import tensorrt as trt  # noqa: N813
    HAS_TRT = True
except ImportError:
    HAS_TRT = False


def export_onnx(model: torch.nn.Module, sample: tuple[torch.Tensor, ...],
                onnx_path: Path, opset: int = 17) -> None:
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            model, sample, str(onnx_path),
            opset_version=opset,
            dynamo=False,           # TorchScript 导出器，成熟稳定
            do_constant_folding=True,
        )


def _cast_gridsample_grid_to_fp16(model_proto) -> None:
    """GridSample 的 input 会被 fp16 化而 grid 常量链可能保留 fp32，TRT 要求二者同型。
    对每个 GridSample 的 grid 输入插一个 Cast(FLOAT16)，保证 input/grid 均为 half。"""
    from onnx import helper, TensorProto

    nodes = model_proto.graph.node
    # 倒序遍历：insert 会移动后续元素，正向遍历会重复处理同一节点（死循环）
    for i in range(len(nodes) - 1, -1, -1):
        node = nodes[i]
        if node.op_type != "GridSample":
            continue
        grid_in = node.input[1]
        cast_out = f"{grid_in}_to_fp16"
        cast = helper.make_node("Cast", [grid_in], [cast_out],
                                to=TensorProto.FLOAT16, name=cast_out)
        del node.input[1]
        node.input.insert(1, cast_out)
        # 拓扑序：Cast 必须位于 GridSample 之前
        nodes.insert(i, cast)


def build_engine(onnx_path: Path, engine_path: Path, fp16: bool = True) -> None:
    import onnx

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    # TRT 10+ 网络默认 explicit batch；TRT 11 移除了 BuilderFlag.FP16，
    # fp16 加速改为在模型侧转换（strongly typed：网络按权重 dtype 计算）
    model_proto = onnx.load(str(onnx_path))
    if fp16:
        from onnxconverter_common import float16
        model_proto = float16.convert_float_to_float16(model_proto, keep_io_types=True)
        _cast_gridsample_grid_to_fp16(model_proto)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(model_proto.SerializeToString()):
        errs = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"ONNX 解析失败:\n{errs}")
    config = builder.create_builder_config()
    # ESRGAN 类 4x 模型在 720p 输入下激活内存即超 5GB，4GB workspace 会静默构建失败
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 12 << 30)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT 引擎构建失败")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(serialized)


class TrtRunner:
    """单个固定 shape 引擎的推理封装。输入/输出为 torch CUDA float32 张量。"""

    def __init__(self, engine_path: Path):
        if not HAS_TRT:
            raise RuntimeError("TensorRT 未安装")
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"引擎加载失败: {engine_path}")
        self.context = self.engine.create_execution_context()
        self.input_names, self.output_names = [], []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)
        # 专用流：避免默认流的每帧隐式同步，让 CPU/GPU 流水线化
        self.stream = torch.cuda.Stream()

    def __call__(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        cur = torch.cuda.current_stream()
        self.stream.wait_stream(cur)  # 等输入张量在当前流上就绪
        # 跨流标记：输入在当前流分配、专用流消费；不标记的话分配器可能在
        # 专用流仍在读时把内存复用给后续分配（当前同步模式下是保险，异步化后是必需）
        for t in inputs:
            t.record_stream(self.stream)
        with torch.cuda.stream(self.stream):
            for name, t in zip(self.input_names, inputs):
                self.context.set_input_shape(name, tuple(t.shape))
                self.context.set_tensor_address(name, t.data_ptr())
            outputs = []
            for name in self.output_names:
                shape = tuple(self.context.get_tensor_shape(name))
                out = torch.empty(shape, dtype=torch.float32, device="cuda")
                out.record_stream(cur)  # 输出在专用流产出、当前流消费
                self.context.set_tensor_address(name, out.data_ptr())
                outputs.append(out)
            ok = self.context.execute_async_v3(self.stream.cuda_stream)
            if not ok:
                raise RuntimeError("TensorRT 执行失败")
        cur.wait_stream(self.stream)  # 输出被当前流消费前等 TRT 完成
        return outputs


def get_engine(model: torch.nn.Module, sample: tuple[torch.Tensor, ...],
               key: str, fp16: bool = True, opset: int = 17) -> TrtRunner:
    """导出(若缺)→构建(若缺)→加载。key 需含模型名与完整输入 shape。"""
    onnx_path = ENGINES_DIR / f"{key}.onnx"
    engine_path = ENGINES_DIR / f"{key}.trt"
    if not engine_path.exists():
        if not onnx_path.exists():
            export_onnx(model, sample, onnx_path, opset=opset)
        build_engine(onnx_path, engine_path, fp16=fp16)
    return TrtRunner(engine_path)
