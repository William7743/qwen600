# Qwen600 HPC 优化题目

本分支 `problem` 是学生题目版。任务是优化 **Qwen/Qwen3-0.6B** 的单 GPU、单请求、batch=1 推理，
在规定数值误差内提高完整请求的性能。原始 V0 推理实现为 `b5c1869`，本分支没有加入参考答案的优化。

**学生使用自己的兼容硬件与软件环境，优化前自行生成本机 V0 logits 参考包。**
优化效果只比较同一硬件和环境中的原始 V0 与候选，不与其他机器的绝对时间直接排名。
最终提交一份包含各优化阶段、思路、验证结果及性能截图的过程文档。

[完整题目与验收规则](PROBLEM.md) · [优化过程文档模板](docs/REPORT_TEMPLATE.md)

## 1. 领取并准备环境

```bash
git clone --branch problem https://github.com/William7743/qwen600.git
cd qwen600
```

使用完整 clone，参考生成需要历史 V0 提交。模型为 **Qwen/Qwen3-0.6B**，使用原始 BF16 safetensors，
指定版本 `c1899de289a04d12100db370d81485cdf75e47ca`，文件哈希见 [模型配置](tests/reference.json)。
权重由学生自行准备，脚本只读取本地文件，不自动下载。不能替换模型、量化权重或修改测试 token 数量。

需要 Linux、支持 BF16 的 NVIDIA Ampere 或更新 GPU、兼容驱动、CUDA Toolkit/cuBLAS/CUB、
C++17 编译器、CMake ≥3.20、PCRE2 8-bit 与 ICU uc 开发库。检查脚本需要 Python 与 NumPy。
现有代码为 CUDA 实现；“自备硬件”指满足这些依赖的设备，不要求与出题方使用同一型号。
既有验证平台为 RTX 3090 / CUDA 12.4 / GCC 8.3.1，这些版本用于参考，不作为统一设备要求。

设置自己的路径与架构，下例 `86` 对应 RTX 3090，其他 GPU 应替换：

```bash
export QWEN_PYTHON=python3
export QWEN_MODEL_DIR=/absolute/path/to/Qwen3-0.6B
export QWEN_CUDA_ARCH=86
cmake -S . -B build-sharegpt -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="$QWEN_CUDA_ARCH" \
  -DQWEN_BUILD_ITERATION=ON -DQWEN_BUILD_BENCHMARKS=ON -DQWEN_BUILD_TESTS=OFF
cmake --build build-sharegpt --target iteration_probe benchmark_probe -j 4
```

若依赖不在系统位置，可为 CMake 添加 `-DCMAKE_PREFIX_PATH=/your/dependency/prefix`；
按需显式设置 CUDA/C++ 编译器。更换工具链使用新构建目录。

## 2. 优化前生成参考，并建立本机基线

**先保持原始 V0 实现不变**，只配置本机工具链和 GPU 架构，然后运行：

```bash
"$QWEN_PYTHON" tests/sharegpt_check.py freeze --model "$QWEN_MODEL_DIR" --build-dir build-sharegpt
"$QWEN_PYTHON" tests/sharegpt_check.py check --model "$QWEN_MODEL_DIR" \
  --build-dir build-sharegpt --output build-sharegpt/check-v0
"$QWEN_PYTHON" benchmarks/run_benchmark.py --model "$QWEN_MODEL_DIR" \
  --build-dir build-sharegpt --warmup 1 --repeats 1 --output build-sharegpt/bench-v0
```

`freeze` 核对 V0 生产源码、重新构建探针，生成固定后续历史与 600 个全词表 logits；
数组约 347.8 MiB，保存在 `build-sharegpt-reference`，并自动写入 `tests/sharegpt_reference.json` 的本机 manifest 哈希。
生成参考时的时间不作为性能基线，性能基线是上面独立执行的 `bench-v0`。

记录参考 manifest 哈希、V0 版本和环境，保留参考包与基线结果。`build-sharegpt-reference` 不是可随意删除的构建缓存。
每位学生生成的参考哈希可能不同；生成后不能修改参考来让候选通过。命令拒绝覆盖非空参考目录。
若换设备或工具链，要在未优化的 V0 中重新建立对应参考及性能基线，并在同一新环境重测候选。

## 3. 开发迭代

每次修改后重新构建；固定六条 ShareGPT 快检用于节省迭代时间：

```bash
cmake --build build-sharegpt --target iteration_probe benchmark_probe -j 4
"$QWEN_PYTHON" tests/sharegpt_check.py check --model "$QWEN_MODEL_DIR" \
  --build-dir build-sharegpt --quick --output build-sharegpt/check-stage1-quick
"$QWEN_PYTHON" benchmarks/run_benchmark.py --model "$QWEN_MODEL_DIR" \
  --build-dir build-sharegpt --quick --warmup 1 --repeats 1 --output build-sharegpt/bench-stage1-quick
```

后续更换输出目录名称。快速结果只能与相同六条负载比较，不能与完整100条基线直接计算加速比。

## 4. 阶段验收与提交

阶段完成后执行完整100条检查与性能测试：

```bash
"$QWEN_PYTHON" tests/sharegpt_check.py check --model "$QWEN_MODEL_DIR" \
  --build-dir build-sharegpt --output build-sharegpt/check-stage1-full
"$QWEN_PYTHON" benchmarks/run_benchmark.py --model "$QWEN_MODEL_DIR" \
  --build-dir build-sharegpt --warmup 1 --repeats 1 --output build-sharegpt/bench-stage1-full
"$QWEN_PYTHON" benchmarks/compare_results.py --v0 build-sharegpt/bench-v0 \
  --v1 build-sharegpt/bench-stage1-full --output build-sharegpt/comparison-stage1.json
```

正式计时为 **protocol 4、ITL 开启、无显存轮询**，不同时运行 profiler 或 sanitizer。
只验收完整模型 logits 与完整请求性能，不要求独立算子通过测试或保留融合前的中间张量。

提交一份 [优化过程文档](docs/REPORT_TEMPLATE.md)，可用 Word、PDF 或含完整截图的 Markdown。
文档包含环境、V0 基线、每个阶段的七步过程、最终汇总和结果截图；不要求额外提交代码仓库、权重或参考数组。
代码版本号、关键代码片段或 diff 可放入文档，便于解释实现。

## 规则与维护入口

数值阈值见 [optimization_policy.json](tests/optimization_policy.json) 的 logits 部分，
正式验收范围见 [acceptance_contract.json](tests/acceptance_contract.json)。
详细指标见 [benchmark 说明](benchmarks/README.md)，计时边界见 [protocol 4 说明](docs/TIMING_PROTOCOL.md)。
旧逐层、独立算子和外部参考测试仅用于可选诊断，见 [维护说明](docs/MAINTAINER_VALIDATION.md)。

本项目借鉴 [yassa9/qwen600](https://github.com/yassa9/qwen600)，保留上游历史与 [MIT 许可](LICENSE)。
