# ShareGPT 简化前的项目说明（历史归档）

以下命令与范围属于旧流程。当前入口见 [项目 README](../../README.md)。

# qwen600.cu

面向 **Qwen3-0.6B BF16 单 GPU、单请求、batch=1** 的 CUDA/C++ 推理项目，
用于学习推理内核、验证数值正确性和考察实习生的 HPC 优化能力。当前不支持请求并发。

## 实习生优化与融合验收

V0 已配套 [快速迭代流程](../FAST_ITERATION.md)。仅增加测试基础设施，不改变 V0 推理实现。
默认 `tests/validate.py` 检查模型输出和 CPU 回归，**不要求旧的独立 CUDA 算子接口**。
快速迭代用 `tests/iterate.py check` / `bench`，失败后用 `diagnose`。
旧算子、逐层和 Transformers 诊断只有显式添加 `--legacy-operators` 才运行。

本机已构建并准备参考包，可运行：

```bash
/home/msganzy/vllm-shared/base-env/bin/python tests/iterate.py check --output build-iteration/check-001
/home/msganzy/vllm-shared/base-env/bin/python tests/validate.py --model /home/msganzy/vllm-shared/models/Qwen3-0.6B --build-dir build-iteration --asan-build-dir build-release-asan --output build-iteration/validation-001
```

全新 clone 先按快速迭代说明构建并准备可信参考包。参考包需在优化前冻结，不能用
实习生的候选输出覆盖；模型权重和大体积参考数组不随仓库发布。下文旧逐层验收
配置保留用于历史复现，不作为融合实现的通用结构要求。

## 当前状态与 V0 / V1

- **V0**：完成必要正确性修复、验收后交给实习生的优化起点，并在其上测量性能基线。
- **V1**：在 V0 上进行 HPC 优化后的版本，必须满足固定误差范围，再比较性能。
  当前尚未实现 V1，也尚未完成正式 V0 性能基线。
- 已修复分词中的缩进/连续空白差异、`<` 和 `top-k=0` 越界、`top-p=1` 概率问题，
  以及 Attention 超过 1024 token 时漏算分数的问题。当前配置总上下文上限为 **8192 token**。
- 加强版独立算子 **287 项检查通过**；模型对照已采集 logits 和逐层 hidden states。
  新增的 192 步相同历史预测中，188 步 top-1 一致，4 步不同。
  **模型相对 Transformers 的完整数值验收仍未完成**，不能将回归通过理解为全面正确。

本机的 `build-v0-reference` 是当前实现的冻结数值输出快照，供优化版对照；
它不是 V1，也不代表正式 V0 已完成验收。该目录被 Git 忽略，没有随仓库发布，
清理构建缓存时应保留。其他机器需要从可信的 V0 结果生成并保管自己的参考快照。

## 环境与构建

需要 Linux、支持 BF16 的 NVIDIA Ampere 或更新 GPU、兼容驱动与 CUDA Toolkit、
cuBLAS/CUB、C++17 编译器、CMake ≥3.20，以及 PCRE2 8-bit 和 ICU uc 开发库。
推理程序不依赖 Python；导出分词器、执行 Transformers 参考测试需要 Python 环境，
内存边界验证还需要 CPU AddressSanitizer。

优先使用已有模型和环境。测试入口不会自动安装依赖或下载权重，版本及模型哈希要求见
[正确性测试说明](../../tests/README.md)。本机使用 RTX 3090 24GB、CUDA Toolkit 12.4，
已有环境的完整构建命令（含独立 ASan 构建）见 [本机验证说明](../LOCAL_VALIDATION.md)。

以下命令均在项目根目录执行；绝对路径是本机示例，其他机器应替换为自己的路径。
构建正确性探针：

```bash
cmake -S . -B build-release-check -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_PREFIX_PATH=/opt/anaconda3 -DQWEN_BUILD_TESTS=ON -DQWEN_ENABLE_ASAN=OFF
cmake --build build-release-check -j 4
```

这里关闭的是主构建中的 CPU ASan；完整验证仍需按本机验证说明准备
`build-release-asan`，不能省略内存检查。

## 正确性测试与优化验收

| 检查对象 | 参考与覆盖 |
| --- | --- |
| 分词 token ID / 聊天模板 | 与 Hugging Face 分词器对照，包含 842 条扩展分词文本 |
| 内存边界、随机采样 | CPU ASan、top-k 边界、top-p 截断与采样频率检查 |
| 独立算子 | 287 项，对照量化输入上的 CPU float64 参考；Attention 覆盖长度 1～8192 |
| 模型 logits | Transformers BF16 参考；新增 33 条用例、296 个位置的全词表向量 |
| hidden states | 新增用例的 embedding、28 层输出及最终归一化，共 1980 个向量 |
| 生成 token ID | 保留三组严格贪心生成回归；新增 192 步相同参考历史预测诊断 |

本机一行运行完整回归（先完成上述构建和独立 ASan 构建）：

```bash
/home/msganzy/vllm-shared/base-env/bin/python tests/validate.py --legacy-operators --model /home/msganzy/vllm-shared/models/Qwen3-0.6B --build-dir build-release-check --asan-build-dir build-release-asan --output build-release-check/validation-strengthened
```

终端和 JSON 报告记录检查结果。`REGRESSION_CHECKS_PASS_NUMERICAL_REVIEW_REQUIRED`
表示约定回归通过、模型数值仍待验收；失败显示 FAIL 并返回非零退出码。
覆盖、参考版本和诊断数据说明见 [tests/README.md](../../tests/README.md)。

以下保留旧逐层验收配置，用于历史复现。当前默认以最终 logits 判定，层输出和独立算子用于诊断；旧配置要求在**每个采样向量上同时满足**：

| 指标 | V1 相对冻结 V0 的上限 |
| --- | --- |
| logits 平均 / 最大绝对误差 | 0.05 / 0.5 |
| softmax 概率总变差 | 0.02 |
| top-1 分数损失 | `max(V0_logits) - V0_logits[argmax(V1_logits)] ≤ 0.125` |
| hidden-state 相对 L2 误差 | 0.01 |
| hidden-state 最大绝对误差 | `0.002 + 0.02 * max(abs(V0))` |
| embedding / 非有限值 | embedding 完全一致；不允许 NaN/Inf |

还必须通过原有回归和全部独立算子检查。这些是本项目的工程验收阈值，
不认证 V0 相对 Transformers 的已有误差，也不保证自由生成全文逐字相同。
完整定义及快照冻结方法见 [优化验收规则](../OPTIMIZATION_ACCEPTANCE.md)。

优化后重新构建测试探针，再追加 V0 对照：

```bash
/home/msganzy/vllm-shared/base-env/bin/python tests/validate.py --legacy-operators --model /home/msganzy/vllm-shared/models/Qwen3-0.6B --build-dir build-release-check --asan-build-dir build-release-asan --output build-release-check/v1-validation --optimization-baseline build-v0-reference
```

## Benchmark

正确性测试集和性能负载集分别维护，性能负载不含标准答案。

| 选择参数 | 请求数 | 输入 / 输出 token 长度 |
| --- | ---: | --- |
| `--dataset diverse100`（默认） | 100 | 内容多样的固定合成输入；两种长度均小于 1024 |
| `--dataset sharegpt100` | 100 | 补充 ShareGPT 子集；两种长度均小于 1024，输出数取原回复的 token 长度 |
| `--dataset fixed9` | 9 | 输入和输出分别取 16、256、1024，组成九种组合 |

长度包含输入聊天模板。集合已保存，运行时不再随机抽样。ShareGPT 是固定前缀筛选
的子集，不代表完整数据分布，来源和许可证见 [子集说明](../../benchmarks/sharegpt100/README.md)。

构建计时程序：

```bash
cmake -S . -B build-benchmark -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_PREFIX_PATH=/opt/anaconda3 -DQWEN_BUILD_BENCHMARKS=ON
cmake --build build-benchmark --target benchmark_probe -j 4
```

默认先执行 **1 次真实请求预热**，再对 100 条负载各测 1 次，最后输出均值：

```bash
/home/msganzy/vllm-shared/base-env/bin/python benchmarks/run_benchmark.py --model /home/msganzy/vllm-shared/models/Qwen3-0.6B --output build-benchmark/results-diverse100
```

补充测量 ShareGPT：

```bash
/home/msganzy/vllm-shared/base-env/bin/python benchmarks/run_benchmark.py --dataset sharegpt100 --model /home/msganzy/vllm-shared/models/Qwen3-0.6B --output build-benchmark/results-sharegpt100
```

结果目录必须为空。模型只加载一次，采用贪心采样、忽略 EOS，严格生成指定数量。
计时不含分词、文本解码、打印和模型加载；包含请求内的数据传输与采样。

- **Prefill**：处理全部输入并使最后一份 logits 在主机可用的耗时。
- **TTFT**：从请求开始到首个输出 token ID 可用，包含 prefill 和首次采样。
- **TPOT**：后续 decode 耗时除以 `输出 token 数 − 1`。
- **总耗时**：TTFT + decode 耗时；另外记录 decode tokens/s、模型加载时间和显存观测峰值。

终端输出逐请求指标及均值；`results.csv` 保存正式测量，`aggregate.json` 保存均值及
按 token 加权的 TPOT/吞吐，`raw.jsonl` 保留原始记录。预热不计入汇总。
单次测量不能判断运行波动；V0/V1 必须使用相同负载、环境及测量规则。
详细计时边界见 [benchmarks/README.md](../../benchmarks/README.md)。当前测量不会被标为正式 V0 成绩。

## 项目来源与致谢 / Attribution

本项目基于并借鉴 [yassa9/qwen600](https://github.com/yassa9/qwen600)，
原作者为 **Yassa Sfen（GitHub: @yassa9）**。当前推理引擎的核心实现来自该上游项目，
包括 CUDA 算子、Transformer 前向计算、权重加载、分词和采样逻辑。

本仓库在原项目基础上整理了源码目录，用于后续学习、正确性修复和性能优化。
保留原项目的 Git 历史、[MIT 许可证](../../LICENSE)及版权声明。感谢原作者的开源贡献。

This repository is based on [yassa9/qwen600](https://github.com/yassa9/qwen600)
by **Yassa Sfen (@yassa9)**. The core inference implementation originates from
that upstream project. This repository reorganizes the source tree for continued
study, correctness fixes, and performance optimization, while preserving the
upstream Git history, MIT license, and copyright notice.

**文档说明：**下文中的第一人称项目介绍、RTX 3050 实验和性能对比来自原项目文档，
代表原作者的实验记录，并非本仓库重新测得的结果。

**Documentation note:** The first-person introduction, RTX 3050 experiments,
and benchmark comparisons below are retained from the upstream documentation
and describe the original author's results, not new measurements by this repository.

## 本仓库的分词与采样修复

分词器按模型的 NFC 规范化、Unicode 正则预分词及有序 BPE 合并对编码，
使用 **ICU（uc）** 和 **PCRE2（8-bit）**。运行推理仍不需要 Python。
`top-k=0` 的全词表选择和 `top-p=1` 的概率归一化已修复。

升级后需要重新编译程序，并重新运行 `tools/export.py <model_dir>`，生成
带 `QTK2` 标识的新 `tokenizer.bin`。旧导出文件不能供新程序使用，旧程序也不能
读取新格式；模型权重无需重新下载。正确性复测步骤见 [tests/README.md](../../tests/README.md)。

## 上游项目介绍与历史实验

以下保留上游介绍及实验记录；本仓库当前构建、验证和计时流程以本文前面的说明为准。

<p align="center">
  <img src="assets/banner.png" width="429" height="139" alt="banner_">
</p>

While studying and practicing  CUDA & GPGPU, thought why not make an inference engine from scratch ? So, chose [QWEN3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) model, small model than can run smoothly on my `RTX 3050 8GB` VRAM.
My intention was (and still) to build educational program to learn about LLMs & transformers while maintaining practice in CUDA programming.

I'm introducing static mini inference engine for `QWEN3-0.6B` instruct model in `bf16`, where its benchmarking claims that it's faster than [llama.cpp](https://github.com/ggml-org/llama.cpp) by approximately `8.5%` & `hf with flash-attn` by `292%` in `tokens/sec`, *see benchmarks below*.

---

What does `qwen600` include:
- single batch inference engine
- static-constanted for compile-time optimization
- all CUDA C/C++, no python dependencies (except for tokenizer setup)
- minimal libraries (cuBLAS, CUB, std IO)
- efficient memory pipeline: mmap, single GPU block, async copy
- zero-cost pointer-based weight management on GPU

---

`qwen600` is inspired by:
- [llama.cpp - ggml](https://github.com/ggml-org/llama.cpp)
- [llama2.c - Andrej Karpathy](https://github.com/karpathy/llama2.c)
- [LLMs-from-scratch - Sebastian Raschka](https://github.com/rasbt/LLMs-from-scratch)
- [qwen3.c - Adrian Cable](https://github.com/adriancable/qwen3.c)

<p align="center">
  <img src="assets/arch.png" width="283" height="401" alt="arch">
</p>

## Design Philosophy

- The design of `qwen600.cu` is heavily inspired by the [suckless philosophy](https://suckless.org/philosophy/).
- The goal is to create a tool that is simple, minimalist, and highly performant by avoiding feature bloat and unnecessary abstractions.
- Configuration is done directly in the source code `config.h` as much as possible, and dependencies are kept to an absolute minimum.

## Project Layout

```text
qwen600/
├── engine/     # CLI and generation loop
├── models/     # Qwen3 transformer and CUDA kernels
├── layers/     # sampling operations
├── utils/      # tokenizer and safetensors loading
├── tools/      # tokenizer export helpers
├── tests/      # correctness probes and V0-relative acceptance
├── benchmarks/ # fixed workloads and timing interface
├── docs/       # validation records and acceptance rules
└── config.h    # compile-time model and runtime constants
```

## WANNA TRY ?!

### Initial Setup

First, you need to clone [QWEN3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B).
This is fantastic [hugging face doc blog](https://huggingface.co/docs/hub/en/repositories-getting-started) to start with cloning hf repos.

then as a safe approach, you locate the weights file (model.safetensors) and sha256sum:
```bash
sha256sum <model_dir>/<safetensors-file-name>
```
and output must be according to hf:
```text
f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b
```

After that:

```bash
git clone git@github.com:William7743/qwen600.git
cd qwen600
```

Assume that downloaded hugging face dir is `<model_dir>`.

We convert the Hugging Face tokenizer into the format used by `qwen600`.

```bash
python tools/export.py <model_dir>
```

That gonna output some template files and most importantly: `tokenizer.bin`

### Building qwen600

Now we are ready to build !
You just want:
- `CUDA` + `nvcc`
- `cuBLAS` + `CUB`
- `PCRE2` (8-bit development headers/library) + `ICU` (`uc` development headers/library)

```bash
mkdir build && cd build
cmake .. && make -j$(nproc)
```

If PCRE2/ICU are installed outside the system paths, set their prefix. On the
current development machine they are available under `/opt/anaconda3`:

```bash
cmake -S . -B build-local -DCMAKE_PREFIX_PATH=/opt/anaconda3
cmake --build build-local -j 4
```
Just that simple, no other bulky libraries and dependencies to build.

## Moment of Truth: Running the Model

You can see arguments manual by:

```bash
# you are now inside qwen600/build
./qwen600
```

the output be that manual:
```aiignore
usage:   ./qwen600 <model_dir> [options]
example: ./qwen600 <model_dir> -r 1
model directory must contain:
  - model.safetensors
  - tokenizer.bin
  - template_*.txt files

arguments:
----------
  -r <int>    reasoning mode, 0 (default) = no thinking, 1 = thinking
  -s <int>    random seed, default
  -k <int>    k value in top-k sampling, default 20
  -t <float>  temperature in [0,inf], default 0.6
  -p <float>  p value in top-p (nucleus) sampling in [0,1], default 0.95
  -i <string> input prompt
  -y <string> system prompt in chat mode, default is none
```

For example:

```bash
./qwen600 <model_dir> -r 1 -t 0.65 -p 0.9 -k 20
```

or simply going with defaults:

```bash
./qwen600 <model_dir> -r 1
```

Based on official hugging face [model card](https://huggingface.co/Qwen/Qwen3-0.6B), they advise that:

```txt
- For thinking mode (enable_thinking=True), use Temperature=0.6, TopP=0.95, TopK=20.
- DO NOT use greedy decoding, as it can lead to performance degradation and endless repetitions.
- For non-thinking mode (enable_thinking=False), we suggest using Temperature=0.7, TopP=0.8, TopK=20, and MinP=0.
```

## Some Experiments

### Without THINKING

```bash
./qwen600 <model_dir> -r 0

>> what is capital of Greece ?
```

> The capital of Greece is **Athens**

> [231.71 tk/s, 19 tokens in 0.08s]

```bash
>> tell me interesting fact about whales ?
```

> One of the most fascinating facts about whales is that they are the only mammals that can swim upside down. This ability allows them to navigate through the ocean in a unique way, using their tails as a natural propeller

> [632.31 tk/s, 227 tokens in 0.36s]

### With the REASONING MODE, aka THINKING

<p align="center">
  <img src="assets/ex01.png" width="577" height="493" alt="ex01_">
</p>

```bash
./qwen600 <model_dir> -r 1

>> what are llms used for ?
```

> Okay, the user is asking what LLMs (Large Language Models) are used for. Let me start by recalling what I know about them. LLMs are a type of AI model that can understand and generate human language. They're used in various fields because they can process a lot of text and perform tasks like answering questions, generating content, and even creating stories.

> First, I should mention their primary applications. Like, in customer service, where they can help with queries. Then, maybe some examples like writing articles, generating code, or even creating content. Also, I should think about how they're used in different industries. For instance, healthcare, finance, education, etc.

> Wait, are there any specific industries where LLMs are particularly used? Maybe in customer service for chatbots, or in creative fields like art or music. Also, in research, maybe for data analysis or information retrieval. I need to make sure the answer is comprehensive but not too technical. Let me structure it step by step: definition, applications, industries, examples, and maybe a brief mention of limitations or challenges.

> I should also check if there's any recent advancements in LLMs that I should include. For example, the development of more powerful models that can handle more complex tasks. But maybe that's beyond the current scope. The user might be looking for a clear and concise answer. Let me make sure to cover key areas without getting too detailed.

now the responding:
> Large Language Models (LLMs) are advanced AI systems designed to understand and generate human language. They are used in a variety of applications across different industries to improve efficiency, accuracy, and accessibility of information. Here are some key uses:

> 1. Customer Service and Chatbots: LLMs help in answering questions, resolving customer issues, and providing real-time assistance. For example, chatbots can handle inquiries, resolve complaints, and even provide product recommendations.

> 2. Writing and Content Creation: They can generate articles, creative content, and even code. For instance, they might assist in writing research papers, creating marketing content, or generating creative scripts.

> 3. Data Analysis and Research: LLMs analyze large datasets, extract insights, and help with information retrieval. They are useful in fields like finance, healthcare, and education for tasks like summarizing data, identifying patterns, and generating reports.

> 4. Creative Industries: In areas like art, music, and design, LLMs can generate original content, music, and visual art, assisting artists and producers.

> 5. Education: They support personalized learning, provide explanations, and assist in language learning by generating interactive content.

> 6. Healthcare: LLMs can help diagnose medical conditions, explain treatments, and support patient education.

> Limitations: While LLMs are powerful, they may face challenges in handling complex tasks, understanding context, or adapting to new information. However, ongoing advancements in training and model architecture continue to expand their capabilities

> [111.44 tk/s, 604 tokens in 5.42s]

## Benchmarking

These benchmarks are done on same machine:
- RTX 3050 8BG + CUDA 13.0
- AMD ryzen 5 3500
- 16GB RAM
- Void Linux

Every Number here is mean of consecutive 5 runs of same parameters where `temp = 0` to be greedy, done manually (no scripts).

Every test is with the same question `what are llms used for ?` in `THINKING` mode.

| inference engine | ~ tokens/sec
| ---              | ---
| hf + flash-attn  | 29.57
| llama.cpp        | 107.19
| qwen600          | **116.15**

`NOTE`: As I mentioned earlier, it is EDUCATIONAL project for me, I'm not aiming for winning a race, but I think that difference caused by static compile-time optimizations, and some other tweaks and tricks.

## TODOs

There are still many catches there:
- [x] Fusing RMSnorm Kernel
- [x] Fusing skip connections with cuBLAS
- [ ] Fix Softmax Kernel & Dispatcher
- [ ] Exploring option of RoPE pre-computed values

## License

MIT
