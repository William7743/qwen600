# Qwen600 HPC 优化题目

本分支 `problem` 是学生题目版。任务是优化 **Qwen/Qwen3-0.6B** 的单 GPU、单请求、batch=1 推理，
在规定数值误差内提高完整请求的性能。原始 V0 推理实现为 `b5c1869`。
正确性验证和性能测试均使用固定的 **ShareGPT100 子集（100 条请求）**；正确性参考输出由本机原始 V0 生成。

**学生使用自己的兼容硬件与软件环境，优化前自行生成本机 V0 logits 参考包。**
优化效果只比较同一硬件和环境中的原始 V0 与候选，不与其他机器的绝对时间直接排名。
最终提交一份包含各优化阶段、思路、验证结果及性能截图的过程文档，结构由学生自行组织。
优化策略由学生自行决定；批处理或多并发改造不计入本题。

## 1. 领取并准备环境

解压项目压缩包，进入包含本 README 的目录即可。
固定 ShareGPT100 已包含在包内，不需要自行下载数据集。运行仍需准备指定模型和兼容环境。

在终端中进入解压后的项目目录，后续命令均在该目录执行。

模型为 **Qwen/Qwen3-0.6B**，使用原始 BF16 safetensors，
指定版本 `c1899de289a04d12100db370d81485cdf75e47ca`，文件哈希见 [模型配置](tests/reference.json)。
权重由学生自行准备，脚本只读取本地文件，不自动下载。不能替换模型、量化权重或修改测试 token 数量。

需要 Linux、支持 BF16 的 NVIDIA Ampere 或更新 GPU、兼容驱动、CUDA Toolkit/cuBLAS/CUB、
C++17 编译器、CMake ≥3.20、PCRE2 8-bit 与 ICU uc 开发库。检查脚本需要 Python 与 NumPy。
现有代码为 CUDA 实现；“自备硬件”指满足这些依赖的设备，不要求与出题方使用同一型号。
既有验证平台为 RTX 3090 / CUDA 12.4 / GCC 8.3.1，这些版本用于参考，不作为统一设备要求。

设置自己的路径与架构，下例 `86` 对应 RTX 3090，其他 GPU 应替换：

```bash
export QWEN_PYTHON=python3
export QWEN_MODEL_DIR=/absolute/path/to/Qwen3-0.6B  # 替换为本机 Qwen3-0.6B 模型文件夹的绝对路径
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

`freeze` 按固定 SHA256 清单核对 V0 生产源码、重新构建探针，生成固定后续历史与 600 个全词表 logits；
数组约 347.8 MiB，保存在 `build-sharegpt-reference`，并自动写入 `tests/sharegpt_reference.json` 的本机 manifest 哈希。
生成参考时的时间不作为性能基线，性能基线是上面独立执行的 `bench-v0`。

`freeze` 使用 [v0_sources.json](tests/v0_sources.json) 校验生产源码，清单来自原始 V0 提交 `b5c1869`，无需 `.git` 或 Git 历史。
任一受检文件缺失或哈希不符即拒绝生成参考；不要修改哈希清单来放行优化后的代码。此检查仅在 `freeze` 时执行，生成参考后可正常修改推理实现并运行 `check`。
首次生成后应保存 `build-sharegpt-reference` 中的 `manifest.json`、`logits.f32`、`histories/` 和生成日志。
`tests/sharegpt_reference.json` 初始未设置 manifest 哈希，由 `freeze` 自动填写；将该哈希记录在过程文档中。
检查时会核对参考 manifest、模型文件、后续历史、logits 数组及误差阈值。


记录参考 manifest 哈希、V0 版本和环境，保留参考包与基线结果。`build-sharegpt-reference` 不是可随意删除的构建缓存。
每位学生生成的参考哈希可能不同；生成后不能修改参考来让候选通过。命令拒绝覆盖非空参考目录。
若换硬件、驱动或工具链，要在未优化的 V0 中重新建立对应参考及性能基线，并在同一新环境重测候选。
文档记录 CPU/GPU、显存容量、驱动、CUDA、编译器、构建模式/架构、Python 版本，以及功耗/频率设置和其他 GPU 任务情况。
编译选项若作为优化项调整，应写明前后差异。

## 3. 验收命令

每个写入报告的优化阶段，以及最终提交版本，都须执行完整 ShareGPT100 正确性验证和性能测试。
同一阶段的正确性结果与性能结果必须来自同一个代码版本，不能用部分请求的结果代替完整测试。
以下以 `stage-01` 为例；各阶段使用不同的输出目录，保留对应结果：

```bash
cmake --build build-sharegpt --target iteration_probe benchmark_probe -j 4
"$QWEN_PYTHON" tests/sharegpt_check.py check --model "$QWEN_MODEL_DIR" \
  --build-dir build-sharegpt --output build-sharegpt/check-stage-01
"$QWEN_PYTHON" benchmarks/run_benchmark.py --model "$QWEN_MODEL_DIR" \
  --build-dir build-sharegpt --warmup 1 --repeats 1 --output build-sharegpt/bench-stage-01
"$QWEN_PYTHON" benchmarks/compare_results.py --v0 build-sharegpt/bench-v0 \
  --v1 build-sharegpt/bench-stage-01 --output build-sharegpt/comparison-stage-01.json
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

各报告阶段和最终验收均使用完整 100 条检查，并与优化前冻结的同一份 V0 参考比较。
验收真实推理路径的最终 logits，不把独立算子或 hidden states 检查作为门槛。
融合、布局变化和消除中间张量均可；算子、逐层或 sanitizer 检查仅供按需诊断，不单独计分。
本题是相对本机 V0 的数值约束，不要求与不同平台逐位相同。
此检查不覆盖分词特殊字符、随机采样、全部上下文边界或全部内存安全情形。

### 性能：完整请求

默认负载为固定 ShareGPT100，输入和输出 token 数量不变。
正式验收使用仓库提供的 benchmark 工具和上述固定参数，不修改计时、统计逻辑或测试负载。
工具实现与参数保持一致；小幅收益应考虑运行波动，不能只挑最好的一次。
正式计时不与 Nsight 或 sanitizer 同跑，profiling 数据只用于分析原因。

主要结果是相同工作负载的总请求耗时与吞吐；同时记录 Prefill、TTFT、TPOT、ITL、
请求总耗时的均值及 P50/P95/P99，以及输入输出合计、全请求输出、Decode、串行请求吞吐。
串行请求吞吐是请求数除以请求计时总和，不表示多并发承载能力。

- 加速比 = V0总请求耗时 / 候选总请求耗时。
- 耗时降低比例 = (V0总请求耗时 − 候选总请求耗时) / V0总请求耗时。
- 区分每阶段相对上一阶段的收益与相对初始V0的累计收益，不能将各项百分比直接相加。

不得修改模型权重、数值阈值、测试输入、输出预算或计时口径来获取成绩。
不得缓存测试用例答案、识别固定用例走捷径，或把必要推理工作搬到计时区间外。
`benchmarks/run_benchmark.py`、`benchmarks/metrics.py` 和 `benchmarks/compare_results.py` 保持固定；探针的修改范围见下文。

### 优化范围与接口

**固定计算任务、正确性标准和计时范围，允许自行调整推理接口。**
允许调整内部算子、线程分工、数据布局、内存管理和融合方式，不要求保留原有 kernel 名称、
独立算子接口或中间张量。接口变化时，可对正确性探针和 benchmark 的调用部分做必要适配，
并在报告中说明修改内容、理由及相关代码差异。

不得改变测试输入、输出数量、参考数据、误差阈值、计时边界、预热与统计规则，
不得将必要的请求处理移出计时区间；各计时边界对应的工作必须实际完成。
正确性验证须覆盖性能测试实际采用的推理路径，并支持在指定位置取得完整 logits 进行对照。
适配不得改变固定 token 历史、检查位置、检查数量或通过条件，也不能另设一条仅供正确性测试通过的计算路径。

## 5. 提交要求

**只需提交一份优化过程文档，结构自由，不提供固定模板。** 可用 Word、PDF 或 Markdown，
不额外要求提交代码仓库、权重或参考数组。

文档应包含硬件软件环境、模型及V0版本、本机参考manifest哈希、未优化基线，
以及各优化阶段的具体改动、优化思路、正确性结果、整体性能变化和最终总结。
可选补充失败或退化方案的处理、实际收益分析和已知限制；也可用代码版本号、关键代码片段或diff辅助说明。

文档附各阶段性优化的性能测试结果截图，展示各阶段性优化前后的整体性能。具体组织与展示方式由学生自行决定。

老师根据文档审查优化思路、实现说明、测量条件与整体收益。
截图与文字属于学生提供的实验记录，不自动等同教师独立复测结论；学生可在本机保留源码和原始结果以便答辩说明。

### 评判标准

正确性是前提，重点考察整体性能收益与优化思路的逻辑性，结合实验记录综合评判。

| 评判维度 | 主要看什么 |
| --- | --- |
| 正确性（前提） | 各报告阶段通过完整 ShareGPT100 正确性验证，遵守接口和测试规则 |
| 整体性能收益（重点） | 同一硬件、环境和负载下，相对原始 V0 的完整请求总耗时加速比；同时关注 TTFT、TPOT 等指标的变化 |
| 优化思路与逻辑（重点） | 能否结合性能分析解释瓶颈、改动理由及实际效果，结论是否得到实验支持 |
| 实验与文档可信度 | 各阶段版本、测量条件和结果对应清楚，截图与结论一致，区分阶段收益与累计收益 |

不按优化数量、代码改动量或单个 kernel 的加速比直接评判；不同硬件之间不比较绝对耗时。

## 6. 文件作用与修改范围

下表中的“可修改”仍须遵守计算任务、数值误差、固定负载与真实计算路径要求。
优化前先用未修改的 V0 生成参考包和性能基线，再修改推理实现。

### 可修改的实现与构建文件

| 文件 | 作用 | 修改范围 |
| --- | --- | --- |
| `models/qwen_model.cuh` | 模型结构、运行状态、CUDA 内核、前向流程和资源管理 | 可修改内部实现及推理接口，也可拆分或新增源码；必要的探针适配须在报告中说明 |
| `utils/static_loader.h` | 读取模型权重、组织权重指针及 GPU 存储 | 可调整加载和内存布局；保留指定模型权重的数值与含义 |
| `layers/sampler.h` | 从 logits 选择 token，包含贪心与随机采样 | 可优化实现；benchmark 使用 `sample_argmax`，须保持贪心选择及并列分数处理规则 |
| `config.h` | 模型维度、数值常量和缓冲区容量等配置 | 可调整实现所需的缓冲区配置或新增调优参数；不能更改模型层数、维度、词表、RoPE 参数等模型定义，容量须满足完整测试负载 |
| `engine/main.cu` | 交互式推理程序入口 | 可修改；默认验收直接调用模型接口，不通过此入口，入口自身的改动不计入 benchmark 收益 |
| `utils/tokenizer.h` | 文本分词、token 解码与提示词模板处理 | 可修改实现并保持分词语义；默认测试读取固定 token ID，不计分词耗时 |
| `CMakeLists.txt`、`cmake/TokenizerDependencies.cmake` | 构建目标、编译参数、链接库与分词依赖查找 | 可适配硬件、添加源码或调整编译选项；保留验收目标、可执行文件位置及真实测试路径，报告编译配置差异 |
| `tests/iteration_probe.cu`、`benchmarks/benchmark_probe.cu` | 正确性与性能测试的模型调用入口 | 仅允许推理接口所需的调用适配；不得改变检查要求、计时边界或统计规则，报告附适配差异说明 |
| 新增推理源码、个人分析脚本或报告 | 承载优化实现和实验记录 | 可以新增；不得替换固定验收逻辑或伪造结果 |
| `.gitignore` | 排除本机构建产物与大文件 | 可补充本地产物规则；不能借此省略报告中需要说明的实现改动 |

### 固定的测试工具、数据与规则

以下文件不属于学生优化范围，不能修改或删除，也不能在构建时绕过其检查。

| 文件或目录 | 作用 |
| --- | --- |
| `tests/sharegpt_check.py` | 生成原始 V0 参考包，或检查候选版 logits |
| `tests/logit_metrics.py` | 计算四项数值误差指标 |
| `tests/optimization_policy.json`、`tests/acceptance_contract.json` | 固定数值阈值和验收范围 |
| `tests/reference.json` | 指定模型版本与文件哈希，附参考环境记录 |
| `tests/v0_sources.json` | 原始 V0 的固定源码 SHA256 清单；参考生成时核对，不可修改 |
| `benchmarks/run_benchmark.py` | 校验负载、组织预热与正式测量、保存结果 |
| `benchmarks/metrics.py`、`benchmarks/compare_results.py` | 汇总性能指标、比较优化前后结果 |
| `benchmarks/sharegpt100/` | 固定的 100 条输入、token ID、回复、请求清单、来源记录与许可；不得自行重新选样或改变输出数量 |
| 各级 `README.md`、`LICENSE` 及数据许可文件 | 题目规则、工具说明、来源与许可；个人补充说明写入自己的报告，不改写题目规则或删去归属信息 |

### 本机生成后固定的参考与结果

| 文件或目录 | 作用 | 使用规则 |
| --- | --- | --- |
| `tests/sharegpt_reference.json` | 记录本机 V0 参考包的 manifest 哈希 | 首次由 `freeze` 自动写入，此后固定；不手工修改以使候选通过 |
| `build-sharegpt-reference/` | 保存 V0 logits、固定 token 历史与参考 manifest | 在未优化的 V0 上生成一次，各阶段复用；不能用候选输出覆盖 |
| `build-sharegpt/bench-v0/` | 保存未优化 V0 的性能基线 | 同一比较环境下保留并复用，不用优化后的结果覆盖 |
| `build-sharegpt/check-stage-*`、`bench-stage-*` 和比较结果 | 保存各阶段正确性及性能证据 | 由工具生成，每阶段使用独立目录；不得手工修改测量值或判定结果 |
| `build-sharegpt/` 中的构建产物 | 编译缓存与可执行文件 | 可重新编译或清理构建产物，但不要误删同目录下保留的基线和阶段结果 |

模型目录中的原始权重与分词器文件同样保持固定。更换比较环境时，按第 2 节重新建立参考与基线。

## 规则与维护入口

数值阈值见 [optimization_policy.json](tests/optimization_policy.json) 的 logits 部分，
正式验收范围见 [acceptance_contract.json](tests/acceptance_contract.json)。
指标字段与输出文件格式见 [benchmark 说明](benchmarks/README.md)。
题目规则、参考包要求与统一测试要求在本 README 维护。
本题目分支保留当前验收所需的文件；数据集构建、计时校准、统计自检等维护工具，以及历史回归、旧报告和合成负载保留在 `main` 分支。
交互程序如需生成分词器文件，可使用 [main 分支的导出工具](https://github.com/William7743/qwen600/blob/main/tools/export.py)；默认 ShareGPT 验收不需要该工具。

本项目借鉴 [yassa9/qwen600](https://github.com/yassa9/qwen600)，保留上游历史与 [MIT 许可](LICENSE)。
