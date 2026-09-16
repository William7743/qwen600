# 正确性验证基线：2026-09-16

## 加强版测试更新

已新增并运行 287 项独立算子检查，全部通过。参考为 BF16 量化输入上的 CPU
float64 计算，覆盖活跃自定义 kernel 和六种实际矩阵乘法形状。Attention 不再只
检查全 1 分数，还对照非均匀数据的分数、概率、V 加权结果以及完整 dispatcher。
8192 长度的独立 Attention 路径经 CUDA memcheck 检查为 0 errors。

模型新增 24 条文本和 9 条长度边界序列，采样 296 个 logits 位置、1980 个逐层
hidden-state 向量；有限值及 embedding 精确一致检查通过。24 条文本各对照 8 步
参考历史下的预测，192 步中 188 步 top-1 相同，4 步差异保留为待定位。
原三组短生成回归结果保持不变。这次没有重跑原有整段 8192 模型对照。

逐层采集是测试探针专用的编译开关，不改变生产 CLI/benchmark 的计算过程。
模型级数值验收仍未完成；不能把独立算子通过等同于整个模型正确。
结果摘要见 [strengthened-summary.json](correctness/strengthened-summary.json)，
用例及误差判定规则见 [tests/README.md](../tests/README.md)。

## 更新：修复 1024 之后的 Attention 分数漏写

`attention_qk_kernel` 改用 `t += blockDim.x` 循环覆盖历史位置，保持
`SEQ_LEN=8192`，无需扩大 CUDA block。当前测试入口上限同步扩展到 8192。

使用本机原有环境、权重和重新构建的探针完成验证：

- 全 1、非均匀 Q/K 两类输入，各 12 个位置，24/24 通过；有效位置与 CPU
  双精度点积参考绝对误差不超过 1e-4，未来位置保持哨兵值。
- CUDA Compute Sanitizer memcheck 检查上述 QK 探针，0 errors。
- 完整模型运行固定 8192 token 序列，11 个采样位置 logits 均有限且 top-1
  与 Transformers 相同。最后位置余弦相似度 0.9998824、MAE 0.11430。
- 分词 842/842、CPU 内存与采样 6/6、三组短生成仍通过。
- 探针拒绝 8193 token 输入；完整数值容差仍待验收，多轮 CLI 不在本次范围。

机器可读结果：[attention-8192.json](correctness/attention-8192.json)。
复现使用 `tests/README.md` 的统一入口；长序列测试会比旧的 1024 测试耗时更长。
以下内容保留旧版验证历史，其 1024 限制描述不代表当前版本。

> **修复后状态：**下文主体保留首次验证的历史结果。分词、越界和采样三类缺陷
> 已修复；随后修复了超过 1024 token 的 QK 分数漏写问题，测试扩展至 8192。
> 复测数据与限制见文末；前向数值差异仍待单独验收。

**结论：整体未通过，尚不能将当前实现视为已验证正确的性能基线。**
3 组短回答与参考逐 token 相同，但确认了分词不一致、长上下文计算错误、
两处内存越界，以及特定采样参数下的概率错误。本次没有修改生产推理代码。

## 环境与方法

- 被测源码：`8dcaccda15f1ce4e32a29217d1d2e6e2bcbfccdb`，新增独立测试文件。
- GPU：RTX 3090 24GB；项目探针：CUDA Toolkit 12.4，`-O3 -arch=sm_86`。
- 参考：Transformers 5.14.1，PyTorch 2.11.0+cu129，BF16，eager attention。
- 两端使用同一本地权重，SHA256：
  `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`。
- 前向对照使用相同参考 token IDs，参考实现也逐 token 运行并保留 KV cache。
- 数值指标包括全词表 logits 的 MAE、最大绝对误差、余弦相似度，
  softmax 概率的总变差距离（TV，0 表示相同，最大为 1），及 top-1/top-10。
- 内存检查采用 AddressSanitizer；采样检查固定种子 42。
- 命令、依赖和测试范围见 [tests/README.md](../tests/README.md)。

## 实测结果

| 检查 | 结果 |
| --- | --- |
| user/system × thinking 开/关模板 | 4/4 完全一致 |
| 分词 token IDs | 10/12 一致；代码和连续空白用例失败 |
| 短提示词逐位置 logits top-1 | 69/73 一致；数值不逐位相同 |
| 3 组短回答贪心生成 | 3/3 完全一致，包含 EOS |
| 参考生成序列的逐步 top-1 对照 | 22/22 一致，包含 EOS |
| attention 单元检查，位置 0/127/1023 | 通过 |
| attention 单元检查，位置 1024/1055 | 失败，分别有 16/512 个分数未正确写入 |
| `top-k=0` | ASan 确认堆缓冲区越界读取 |
| 孤立 `<` 输入 | ASan 确认堆缓冲区越界读取 |
| `top-p=1`，两个等概率候选 | 失败，2000 次得到 505/1495 |
| `top-p=0.95`，同样的候选与种子 | 对照通过，2000 次得到 1031/969 |

### 1. 分词：相同文本可以产生不同 token IDs

代码用例为 `def add(a, b):\n    return a + b\n`。
缩进附近本项目产生 `[257, 689]`，参考产生 `[262, 470]`。
空白用例为 `hello   world\n\n\tend  `，本项目与参考同样产生不同 IDs。
两端都能解码回原文，因此“解码回来一样”不能证明分词正确。

本地 `tokenizer.json` 定义了正则预分词，再执行 ByteLevel/BPE；
`utils/tokenizer.h` 则从整段文本的字节开始全局合并，没有实现相同的预分词边界。
这解释了连续空白和代码缩进附近的差异。完整 IDs 保存在 JSON 报告中。

### 2. Attention：从第 1025 个 token 起存在确定的计算缺失

`models/qwen_model.cuh` 的 QK kernel 以 `threadIdx.x` 直接作为历史位置，
dispatcher 每个头最多启动 1024 个线程，kernel 没有循环覆盖后续位置。
而 softmax 和 V 聚合仍处理全部历史位置。

用全 1 的 Q/K 验证时，每个分数都应为 `sqrt(128)`；
零基位置 1024 时每个头缺 1 项，16 个头共缺 16 项。
位置 1055 时每个头缺 32 项，共缺 512 项。

相同固定 token 序列的完整模型对照也显示边界处显著失真：

| 已输入 token 数 | logits MAE | logits 余弦相似度 | 概率 TV | top-1 相同 |
| --- | --- | --- | --- | --- |
| 1024 | 0.04223 | 0.999950 | 0.00000434 | 是 |
| 1025 | 1.49605 | 0.894860 | 0.999345 | 否 |
| 1056 | 2.55296 | 0.329644 | 0.990975 | 否 |

这些具体误差值对应本次重复英文 token 序列，不能视为所有输入的统一误差。
探针将 attention 暂存区初始清零以便复现；原 CLI 的未初始化数据可能使结果更不稳定。

### 3. 采样：`top-k=0` 越界，`top-p=1` 分布错误

`build_sampler()` 将 `top-k=0` 转为词表大小，随后调用
`quick_select(..., n_cands)`，其中 pivot 为 `arr[k]`。
当 `k == VOCAB_SIZE` 时访问数组末尾之外，ASan 已复现。

概率归一化后 `prob_sum` 仍保留归一化前的总和；只有进入 nucleus 分支时才被覆盖。
当 `top-p=1` 时不进入该分支，随机数仍乘以旧总和，超出累计分布后落到最后一个候选。
两个相同 logits 的候选理论上各 50%，实际分别为 25.25% 和 74.75%。
`top-p=0.95` 的同条件对照为 51.55% 和 48.45%，支持问题来自该分支的归一化处理。

### 4. 输入边界：`<` 扫描越过字符串末尾

`utils/tokenizer.h:121` 的循环检查 `*c != 0`，却读取 `c[k]`。
对于以 `<` 结尾、后面没有 `>` 的文本，无法在字符串终止位置停止。
使用仅包含 `<` 和结尾 NUL 的两字节堆缓冲区，ASan 在第 122 行确认越界读取。

### 5. 短输入数值差异：仍需审阅，不能直接宣布完全正确

三组短提示词共 73 个位置，top-1 相同 69 个。
其中三个不一致位置是共同模板前缀，参考 top-1 与 top-2 本就并列；
另一个位置参考分差为 0.125。三个提示词最终生成的回答均逐 token 相同：

- 英文：`The capital of France is Paris.`
- 中文：`中国的首都是北京。`
- 算术：`2 + 3 = 5`

短提示词的最大绝对 logits 误差为 1.38672，最低余弦相似度为 0.996626，
最大概率 TV 为 0.131174。这些是实测诊断值，不是验收阈值。

源码可见多处精度路径差异：RMSNorm 的权重乘法前是否回写 BF16、
RoPE 三角函数及乘加的舍入、Attention 概率是否先转 BF16、SwiGLU 的中间舍入，
以及 cuBLAS 残差融合。它们可能解释部分数值差异，但本次未逐算子或逐层归因，
因此不能断言全部差异都来自无害舍入，也没有设置事后放宽的“通过阈值”。

## 后续顺序

先修复 attention 覆盖范围、两处越界和采样归一化，再对齐分词规则；
之后重跑这些用例，并增加逐层数值定位，制定明确的数值验收标准。
正式 benchmark 应以修复并通过约定正确性检查的版本为基线。

本次不是全面认证：没有覆盖所有 Unicode、所有采样参数、多轮 chat 循环，
也没有完成 8192 长度的正确性证明。

## 结果文件

- [完整数值及分词报告](correctness/baseline-2026-09-16.json)
- [内存及采样检查摘要](correctness/edges-2026-09-16.json)
- 本机 `build-correctness/results/` 保留原始 logits、输入 IDs 和 ASan 日志。

构建和原始大文件由现有 `build-*/` 忽略规则排除，不包含模型权重。

## 修复后复测

### 改动

- `tools/export.py` 导出 QTK2 格式，保存模型的 NFC 开关、正则预分词规则、
  256 个字节 token IDs、注册的 added tokens，以及精确的 BPE 合并对和顺序。
- `utils/tokenizer.h` 使用 ICU NFC 和 PCRE2 Unicode 正则，限制 BPE 合并在
  各预分词片段内；仅匹配注册的 added tokens，移除无界 `<` 扫描。
- 分词输出改为动态 vector，覆盖 NFC 可能增加字节长度的字符；读取导出文件时
  校验魔数、长度、IDs 和合并结果。旧导出格式会被明确拒绝。
- `layers/sampler.h` 在选择全词表时跳过 quickselect，其余使用零基索引 `k-1`；
  概率归一化后重置概率总和，并在 nucleus 累计概率达到阈值时停止。
- 不修改 `models/qwen_model.cuh` 或 `config.h`；验证范围继续限制在最多 1024 token。

分词流程依据模型本地 tokenizer.json；相关实现语义参见
[Hugging Face 分词流水线](https://www.huggingface.co/docs/tokenizers/python/latest/pipeline.html)
及 [PCRE2 Unicode 支持](https://pcre.org/current/doc/html/pcre2unicode.html)。

### 结果

| 检查 | 修复后结果 |
| --- | --- |
| 原有分词测试 | 12/12 完全一致 |
| 模板 | 4/4 完全一致 |
| 扩展分词用例 | 普通构建与 ASan 构建均为 842/842，与 AutoTokenizer 的 token IDs 一致 |
| `<` 与 `top-k=0` 的 ASan 检查 | 均通过，无原先的越界报告 |
| `top-p=1` 等概率采样 | 969/1031，符合预设容差 |
| `top-p=0.95` 同条件对照 | 969/1031，通过 |
| 扩展 top-k 边界及 nucleus 恰好达到阈值 | 均通过 |
| 三组短回答 | 3/3 逐 token 一致 |
| 最多 1024 token 的 attention 检查 | 全部通过 |
| 前向数值对照 | 仍为 69/73 个短提示词位置 top-1 一致，差异未改变 |

模型测试退出码为 0，报告状态为
`STRUCTURAL_CHECKS_PASS_NUMERICAL_REVIEW_REQUIRED`。
这说明本轮确定性缺陷的回归检查通过，不表示所有前向数值差异已验收，
也不表示模型支持超过 1024 token 的正确计算。

构建已通过，CLI 短文本推理正常。新增的原生依赖为 PCRE2 8-bit 和 ICU uc，
本机使用 `/opt/anaconda3` 中已有的库。`tokenizer.bin` 已重新导出为 QTK2；
使用者必须重新编译旧程序，不能用旧二进制读取新格式。

复测命令见 [tests/README.md](../tests/README.md)，结果如下：

- [修复后模型与分词对照](correctness/fixed-2026-09-16.json)
- [修复后采样与内存检查](correctness/edges-fixed-2026-09-16.json)
- [扩展分词检查](correctness/tokenizer-fixed-2026-09-16.json)

本地构建、日志和原始 logits 位于 `build-correctness-fixed/`，未改变首次基线文件。
