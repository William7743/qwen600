# Qwen600 HPC 优化题目

本分支 `problem` 是学生题目版。任务是优化 **Qwen/Qwen3-0.6B** 的单 GPU、单请求、batch=1 推理，
在规定数值误差内提高完整请求的性能。原始 V0 推理实现为 `b5c1869`。
正确性验证和性能测试均使用固定的 **ShareGPT100 子集（100 条请求）**；正确性参考输出由本机原始 V0 生成。

**学生使用自己的兼容硬件与软件环境，优化前自行生成本机 V0 logits 参考包。**
优化效果只比较同一硬件和环境中的原始 V0 与候选，不与其他机器的绝对时间直接排名。
最终提交一份包含各优化阶段、思路、验证结果及性能截图的过程文档，结构由学生自行组织。
优化策略由学生自行决定；批处理或多并发改造不计入本题。

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
  -DQWEN_BUILD_ITERATION=ON -DQWEN_BUILD_BENCHMARKS=ON
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

`freeze` 要求生产源码与原始 V0 提交 `b5c1869` 一致，拒绝用优化后的实现生成参考。
首次生成后应保存 `build-sharegpt-reference` 中的 `manifest.json`、`logits.f32`、`histories/` 和生成日志。
`tests/sharegpt_reference.json` 初始未设置 manifest 哈希，由 `freeze` 自动填写；将该哈希记录在过程文档中。
检查时会核对参考 manifest、模型文件、后续历史、logits 数组及误差阈值。


记录参考 manifest 哈希、V0 版本和环境，保留参考包与基线结果。`build-sharegpt-reference` 不是可随意删除的构建缓存。
每位学生生成的参考哈希可能不同；生成后不能修改参考来让候选通过。命令拒绝覆盖非空参考目录。
若换硬件、驱动或工具链，要在未优化的 V0 中重新建立对应参考及性能基线，并在同一新环境重测候选。
文档记录 CPU/GPU、显存容量、驱动、CUDA、编译器、构建模式/架构、Python 版本，以及功耗/频率设置和其他 GPU 任务情况。
编译选项若作为优化项调整，应写明前后差异。

## 3. 验收命令

提交前重新构建优化版本，执行完整100条检查与性能测试：

```bash
cmake --build build-sharegpt --target iteration_probe benchmark_probe -j 4
"$QWEN_PYTHON" tests/sharegpt_check.py check --model "$QWEN_MODEL_DIR" \
  --build-dir build-sharegpt --output build-sharegpt/check-final
"$QWEN_PYTHON" benchmarks/run_benchmark.py --model "$QWEN_MODEL_DIR" \
  --build-dir build-sharegpt --warmup 1 --repeats 1 --output build-sharegpt/bench-final
"$QWEN_PYTHON" benchmarks/compare_results.py --v0 build-sharegpt/bench-v0 \
  --v1 build-sharegpt/bench-final --output build-sharegpt/comparison-final.json
```

正式验收统一使用上述命令和仓库提供的测试工具，运行时不同时开启 profiler 或 sanitizer。
只验收完整模型 logits 与完整请求性能，不要求独立算子通过测试或保留融合前的中间张量。

## 4. 验收规则

### 正确性：完整模型输出

默认使用 100 条固定 ShareGPT，在每条 6 个选定位置比较完整 151936 维 logits，共 600 个向量。
各版本使用相同 prompt 和本机 V0 冻结的后续 token 历史，不采用候选自己的续写替代参考历史。
原 ShareGPT 回复只确定输出数量 G；参考历史由 prompt 和本机 V0 贪心生成的前 G−1 个 token 组成。
设输入长度为 P，六个零基采样位置为 `{0, (P−1)//2, P−1, P, P+(G−2)//2, P+G−2}`。

每个位置必须无 NaN/Inf，并同时满足：

| 指标 | 上限 |
| --- | ---: |
| logits 平均绝对误差 | 0.05 |
| logits 最大绝对误差 | 0.5 |
| softmax 概率总变差 TV | 0.02 |
| V0 对候选 top-1 的分数损失 | 0.125 |

最终验收使用完整100条检查。
验收真实推理路径的最终 logits，不把独立算子或 hidden states 检查作为门槛。
融合、布局变化和消除中间张量均可；算子、逐层或 sanitizer 检查仅供按需诊断，不单独计分。
本题是相对本机 V0 的数值约束，不要求与不同平台逐位相同。
此检查不覆盖分词特殊字符、随机采样、全部上下文边界或全部内存安全情形。

### 性能：完整请求

默认负载为固定 ShareGPT100，输入和输出 token 数量不变。
正式验收使用仓库提供的 benchmark 工具和上述固定参数，不修改计时、统计逻辑或测试负载。
工具实现与参数保持一致；小幅收益应考虑运行波动，不能只挑最好的一次。
正式计时不与 Nsight 或 sanitizer 同跑，profiling 数据只用于分析原因。
fixed9 可辅助研究长度敏感性，不是额外必交集合。

主要结果是相同工作负载的总请求耗时与吞吐；同时记录 Prefill、TTFT、TPOT、ITL、
请求总耗时的均值及 P50/P95/P99，以及输入输出合计、全请求输出、Decode、串行请求吞吐。
串行请求吞吐是请求数除以请求计时总和，不表示多并发承载能力。

- 加速比 = V0总请求耗时 / 候选总请求耗时。
- 耗时降低比例 = (V0总请求耗时 − 候选总请求耗时) / V0总请求耗时。
- 区分每阶段相对上一阶段的收益与相对初始V0的累计收益，不能将各项百分比直接相加。

不得修改模型权重、数值阈值、测试输入、输出预算或计时口径来获取成绩。
不得缓存测试用例答案、识别固定用例走捷径，或把必要推理工作搬到计时区间外。
优化后的推理实现须与现有验收探针兼容，并执行真实计算路径。
`benchmarks/` 下的 `benchmark_probe.cu`、`run_benchmark.py`、`metrics.py` 和 `compare_results.py` 为统一测量工具，不属于本题修改范围。

## 5. 提交要求

**只需提交一份优化过程文档，结构自由，不提供固定模板。** 可用 Word、PDF 或 Markdown，
不额外要求提交代码仓库、权重或参考数组。

文档应包含硬件软件环境、模型及V0版本、本机参考manifest哈希、未优化基线，
以及各优化阶段的具体改动、优化思路、正确性结果、整体性能变化和最终总结。
可选补充失败或退化方案的处理、实际收益分析和已知限制；也可用代码版本号、关键代码片段或diff辅助说明。

文档附各阶段性优化的性能测试结果截图，展示各阶段性优化前后的整体性能。具体组织与展示方式由学生自行决定。

老师根据文档审查优化思路、实现说明、测量条件与整体收益。
截图与文字属于学生提供的实验记录，不自动等同教师独立复测结论；学生可在本机保留源码和原始结果以便答辩说明。

## 规则与维护入口

数值阈值见 [optimization_policy.json](tests/optimization_policy.json) 的 logits 部分，
正式验收范围见 [acceptance_contract.json](tests/acceptance_contract.json)。
指标字段与输出文件格式见 [benchmark 说明](benchmarks/README.md)。
题目规则、参考包要求与统一测试要求在本 README 维护。
本题目分支保留当前验收所需的文件；历史回归工具、旧报告和合成负载保留在 `main` 分支。

本项目借鉴 [yassa9/qwen600](https://github.com/yassa9/qwen600)，保留上游历史与 [MIT 许可](LICENSE)。
