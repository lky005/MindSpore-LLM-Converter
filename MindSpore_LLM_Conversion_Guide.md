# MindSpore Lite 端侧大语言模型云端转换指南 (OpenHarmony)

本文档记录了在 OpenHarmony 6.0 (API 22) 环境下，利用 GitHub Actions 云端资源，将 HuggingFace 开源大语言模型（如 Qwen2.5）转换为华为手机 NPU 硬件加速专用的 `.ms` 格式踩过的坑与终极解决方案。

## 🎯 方案背景与初衷
端侧大语言模型（LLM）的量化和编译非常吃电脑内存，且环境配置（WSL、Python、CANN 驱动等）极其繁琐。通过编写 GitHub Actions 工作流脚本，我们可以白嫖微软的高性能云端 Ubuntu 虚拟机，实现从“自动下载 ONNX 权重 -> W4A16 极速量化 -> 产出 .ms 文件”的无人值守全链路流水线。

## 🚧 踩坑记录与解决方案 (Troubleshooting)

在调试编译 Qwen 2.5 纯文本模型时，我们经历了以下几次典型的编译崩溃，特此记录标准解法：

### 1. HuggingFace CLI 下载工具兼容性问题
*   **报错现象**：`Warning: huggingface-cli is deprecated... ls: cannot access './hf_model/onnx': No such file or directory`
*   **问题原因**：新版 `hf download` 不支持像之前那样通过空格分隔传入多个文件正则匹配模式（如 `--include "a" "b"` 会直接跳过下载）。
*   **正确解法**：必须多次使用 `--include` 参数显式声明每个匹配模式，例如：
    ```bash
    hf download [repo] --include "onnx/model.*" --include "tokenizer*" --include "*.json" --local-dir ./hf_model
    ```

### 2. MindSpore Lite 2.3.0 量化参数废弃
*   **报错现象**：`bitNum is not a valid flag... Flags PreInit failed. Ret: -600`
*   **问题原因**：在 MindSpore Lite 2.3.0 最新版中，官方直接砍掉了命令行里的 `--bitNum` 或 `--weightQuantWeight` 传参方式。
*   **正确解法**：必须在工作流中动态生成 `.cfg` 配置文件，并通过 `--configFile=quant.cfg` 传递给编译器。
    ```ini
    # quant.cfg 内容示例：
    [common_quant_param]
    quant_type=WEIGHT_QUANT
    bit_num=4
    ```

### 3. Ubuntu 服务器缺少华为 Ascend CANN 驱动
*   **报错现象**：`Get real path of libascend_pass_plugin.so failed... Convert model failed`
*   **问题原因**：为了让手机满血跑 NPU，我们给 `converter_lite` 加了 `--optimize=ascend_oriented` 参数，但 GitHub Ubuntu 虚拟机上并没有安装昇腾显卡驱动。
*   **正确解法（JIT 动态映射机制）**：将编译期参数改为 `--optimize=general`。OpenHarmony 的 NNRt（神经网络运行时）会在真机上首次加载该 `.ms` 模型时，以 **JIT (即时编译)** 的形式自动把算子映射给麒麟 SoC 底层的 NPU/GPU 执行。

### 4. 动态序列长度 (KV Cache) 导致 Shape 推断失败 (最深坑)
*   **报错现象**：`InferShapeByNNACL for op: /model/ScatterND failed... Shape is empty`
*   **问题原因**：LLM 对话文本长度不固定（KV Cache 需要动态轴 `dynamic_dims`）。普通的图优化器要求输入 Tensor 的 Shape 是完全写死（Static）的，这导致碰到注意力机制的 `ScatterND` 算子时直接崩溃。
*   **正确解法**：必须在 `converter_lite` 中追加隐藏参数 `--optimizeTransformer=true`。这个参数会告知编译器当前是一张基于 Transformer 架构的语言大模型图，自动开启动态图算子融合并忽略长度推断失败。

### 5. 模型选择：为什么不用最新的 Qwen 3.5？
*   **经验总结**：阿里最新发布的 Qwen 3.5 全系默认采用原生多模态架构（Native Multimodal），即使是 0.5B 或 2B 这种小模型，ONNX 导出结构中也强制携带了 `vision_encoder.onnx`。
*   **结论**：多模态混合网络直接放入 MindSpore Lite 转换必定报算子不兼容错误。对于 16G 内存的高端设备而言，采用最成熟的纯文本网络架构 **`Qwen2.5-1.5B-Instruct`**（W4A16 压缩后仅约 1 GB）或更小的 **`Qwen2.5-0.5B-Instruct`** 进行快速测试，是目前端侧开发最完美且最稳定的平替方案。

---

## 🚀 成果验收与下一步
基于以上修正，当前的 `.github/workflows/mindspore_llm_converter.yml` 已经是一套**身经百战的完美模版**。
编译产出的 `.zip` 压缩包内包含了 `qwen2.5_0.5b_instruct.ms` 与 `tokenizer.json`，可直接解压部署到 OpenHarmony C++ 工程的 `rawfile/model/` 目录下，交由 NAPI 与底层的 NNRt 引擎执行本地推理。