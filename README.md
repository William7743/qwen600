# qwen600 — HPC 内核优化练习

基于 Qwen3-0.6B BF16 的 CUDA/C++ 推理项目，支持单 GPU、单请求、batch=1。
实习生从冻结 V0 推理实现 `b5c1869` 开始优化，满足固定 logits 误差约定后比较性能。
本项目借鉴 [yassa9/qwen600](https://github.com/yassa9/qwen600)，保留 MIT 许可及上游来源。

## 环境与构建

需要 Linux、支持 BF16 的 NVIDIA Ampere 或更新 GPU、兼容驱动、CUDA Toolkit/nvcc、
cuBLAS/CUB、C++17 编译器、CMake ≥3.20、PCRE2 8-bit 和 ICU uc 开发库。
检查脚本需要 Python 与 NumPy；benchmark 只使用 Python 标准库。
已验证平台为 RTX 3090、CUDA 12.4、GCC 8.3.1。权重与参考数组不随 Git 发布，不自动下载。

以下从仓库根目录执行，按机器调整本地路径：

```bash
export QWEN_PYTHON=/home/msganzy/vllm-shared/base-env/bin/python
export QWEN_MODEL_DIR=/home/msganzy/vllm-shared/models/Qwen3-0.6B
cmake -S . -B build-sharegpt -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_PREFIX_PATH=/opt/anaconda3 -DQWEN_BUILD_ITERATION=ON -DQWEN_BUILD_BENCHMARKS=ON -DQWEN_BUILD_TESTS=OFF
cmake --build build-sharegpt --target iteration_probe benchmark_probe -j 4
```

必要时为新构建目录指定 `-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc`、
`-DCMAKE_CXX_COMPILER=/opt/rh/devtoolset-8/root/usr/bin/g++` 与
`-DCMAKE_CUDA_HOST_COMPILER=/opt/rh/devtoolset-8/root/usr/bin/g++`。
模型为本地 Qwen3-0.6B，哈希见 [reference.json](tests/reference.json)。

## 日常优化：两条命令

默认正确性与性能测试共用 **100 条固定 ShareGPT 输入**。修改内核后先重新构建：

```bash
cmake --build build-sharegpt --target iteration_probe benchmark_probe -j 4
```

然后检查 logits，通过后测量性能（每次使用新的输出目录）：

```bash
"$QWEN_PYTHON" tests/sharegpt_check.py check --model "$QWEN_MODEL_DIR" --output build-sharegpt/check-001
"$QWEN_PYTHON" benchmarks/run_benchmark.py --model "$QWEN_MODEL_DIR" --output build-sharegpt/bench-001
```

正确性默认检查全部 100 条；加 `--quick` 可只跑其中固定的 6 条，用于开发迭代。
性能也可加 `--quick` 使用相同 6 条。默认一次真实请求预热，再每条测一次；`--case s001` 可选单条，`--repeats 3` 可重复测量。
快速检查不能代替最终的 100 条验收。

### 正确性检查什么

使用相同权重、相同 prompt 和冻结 V0 的后续 token 历史，在每条的 6 个固定位置比较
全部 **151936 维 logits**，共 600 个向量。覆盖 prompt 首/中/末位置及 decode 首/中/末位置。
候选逐步读入参考历史，不用自己的预测改写后续输入。ShareGPT 原回复只确定输出数量，
不充当模型标准答案。算子融合不需要保留旧算子接口或中间张量。

每个位置同时满足以下条件才 PASS；出现 NaN/Inf 或任一超限则 FAIL，退出码非零：

| 误差指标 | 上限 |
| --- | ---: |
| logits 平均绝对误差 | 0.05 |
| logits 最大绝对误差 | 0.5 |
| softmax 概率总变差 TV | 0.02 |
| V0 对候选 top-1 的分数损失 | 0.125 |

阈值来自 [optimization_policy.json](tests/optimization_policy.json) 的 logits 部分。
输出目录包含 `report.json`、实际 logits 和探针日志。PASS 表示这些采样位置符合 V0 相对误差约定，
不表示所有输入都已验证，也不认证 V0 与 Transformers 完全一致。

参考包由出题方冻结并提供，默认放在 `build-sharegpt-reference`；也可用 `--baseline /path/to/reference`。
Git 中的 [sharegpt_reference.json](tests/sharegpt_reference.json) 固定参考包 manifest 的 SHA256。
**全新 clone 不包含大体积参考数组**，开始优化前请取得参考包；不能用候选实现重建答案。
不要删除参考包或修改阈值、输入、参考哈希来取得 PASS。

### Benchmark 记录什么

| 类别 | 指标 |
| --- | --- |
| 延迟 | Prefill、TTFT、TPOT、请求总耗时、逐 token ITL |
| 延迟统计 | 均值及 P50 / P95 / P99；请求指标按请求统计，ITL 按实际 token 间隔统计 |
| 吞吐 | 输入与输出合计 token/s、全请求输出 token/s、decode token/s、串行请求/s |
| 资源 | 默认不测显存；可选独立 `--memory-only` 运行，模型加载时间另记 |

本项目 **batch=1、并发请求数=1**。串行请求吞吐 = 请求数 / 各请求计时总和，表示当前负载下
引擎逐条完成请求的速度，不能解释为并发服务承载能力。总 token 吞吐和全请求输出吞吐使用
相同时间分母；decode 吞吐只用 decode 时间。

计时包含请求内传输与采样，排除模型加载、分词、文件读写、打印和请求间空隙。
ITL 是主机上相邻 token ID 可用的间隔，不是网络流式响应间隔。
结果保存于 `raw.jsonl`、`results.csv`、`itl.csv`、`aggregate.json` 和 `summary.json`。
P99 是当前有限样本的分位数；不同长度请求间的差异不等于同一请求的重复运行波动。
当前计时为 protocol 4，默认开启 ITL、不采显存。详细公式见 [benchmark 说明](benchmarks/README.md)，
边界及开销校准见 [计时方法](docs/TIMING_PROTOCOL.md)。

## 维护者工具与历史记录

旧 34 条 / 307 个 logits、8192 长历史、分词、采样、独立算子和 sanitizer 流程均保留，
不进入实习生默认验收。原 `fixed9`、`diverse100` 也保留用于历史复现，不作为新的默认负载。
准备参考包及按需诊断见 [维护者说明](docs/MAINTAINER_VALIDATION.md)。
输入/输出长度、来源与许可证见 [ShareGPT 子集说明](benchmarks/sharegpt100/README.md)。

本次工具验证记录见 [ShareGPT 验证记录](docs/sharegpt-validation/README.md)。

## 文件导航

| 目录 | 内容 |
| --- | --- |
| `models/`、`config.h` | 模型前向、CUDA 内核与固定尺寸 |
| `engine/`、`layers/`、`utils/` | CLI、采样、分词与权重读取 |
| `tests/sharegpt_check.py` | 默认 logits 验收 |
| `benchmarks/` | ShareGPT 数据、计时器与汇总 |
| `docs/` | 维护说明与历史记录 |

旧 CLI 用法及完整项目说明保留在 [历史归档](docs/history/README_before_sharegpt.md)。
V0 与 Transformers 的完整外部数值认证尚未完成；工程相对误差验收与模型答案质量是不同问题。
