# 实习生优化的数值验收规则（版本 1）

本规则验收 **优化后的 V1 与冻结 V0 的数值等价性**。V0 相对 Transformers 的
已有误差不因此被认证正确，也不能被当作 V1 继续增加误差的预算。两边使用相同
权重、输入 token、参考生成历史和检查位置，保持单请求 BF16 推理。

规则由维护者固定在 [optimization_policy.json](../tests/optimization_policy.json)。
这是本项目选择的工程阈值，不是行业通用 BF16 误差标准。测试和参考快照应由
出题方保管，学生只修改允许的实现代码，不能修改阈值、用例或参考输出来取得通过。

## 所有条件必须同时满足

| 对象 | 判定条件 |
| --- | --- |
| 每个已采样的全词表 logits 向量 | 平均绝对误差 ≤0.05，最大绝对误差 ≤0.5 |
| 对应 softmax 概率 | 总变差 `0.5 * sum(abs(p_V0-p_V1))` ≤0.02 |
| 候选 top-1 | `max(V0_logits) - V0_logits[argmax(V1_logits)]` ≤0.125 |
| 每个已采样 hidden-state 向量 | `norm(V1-V0)/max(norm(V0),1e-30)` ≤0.01 |
| hidden-state 最大绝对误差 | ≤`0.002 + 0.02 * max(abs(V0))` |
| embedding | 完全一致 |
| 所有参与对照的数据 | 有限，无 NaN/Inf |
| 独立算子 | 287 项检查全部通过，覆盖和原有阈值不得改变 |

这里的 logits 平均误差是在**单个位置的词表维度**上平均，不是把多个请求混在一起
平均。任何一个采样位置超限都判 FAIL，不能以其他位置的较小误差抵消。
hidden-state 的 1% 是向量范数比例，不代表允许每个元素有 1% 的误差。

top-1 规则允许真正接近的候选更换顺序；如果 V0 第一名领先第二名超过 0.125，
V1 必须保持该位置 top-1。这并不允许预设整体 1% 或 5% 的 token 错误率；每一个
发生变化的位置都必须满足分数差、logits 和概率分布限制。允许近似并列也意味着
不保证自由生成全文逐字相同，相同历史下的对照用于防止后续分歧放大。

## 冻结一次 V0

先在可信 V0 上运行加强版测试，保留 `extended-model-report.json`、
`operators-report.json`、`*.forward.ids`、`*.trace.ids` 和 `*.native.f32`。
本机已经把本次结果冻结在 `build-v0-reference`；它是被 Git 忽略的本地输出快照，
不要在清理构建目录时删除。没有将大体积浮点数组推送到 GitHub。

其他环境可从自己的可信 V0 结果冻结：

```bash
python tests/check_optimization.py freeze --results /path/to/v0/results --baseline /path/to/frozen-v0
```

冻结目录必须为空；`baseline.json` 保存每个文件的 SHA256 以及固定阈值。出题方
应将整个目录及 manifest 哈希另行留存。冻结函数不会修复或认证 V0 的既有数值问题。

## 验收 V1

重新构建优化版探针后，在项目根目录运行全部测试并追加 V0 对照：

```bash
/home/msganzy/vllm-shared/base-env/bin/python tests/validate.py --model /home/msganzy/vllm-shared/models/Qwen3-0.6B --build-dir build-release-check --asan-build-dir build-release-asan --output build-release-check/v1-validation --optimization-baseline build-v0-reference
```

如果已经生成 V1 的加强版结果，可以只运行数值等价性对照：

```bash
python tests/check_optimization.py compare --baseline build-v0-reference --candidate /path/to/v1/results --output /path/to/optimization-report.json
```

比较器检查基线文件哈希、固定阈值、输入/历史 ID、位置与算子覆盖，再读取实际
浮点数组重新计算误差，不直接相信候选报告里的误差统计。会输出每条用例的
PASS/FAIL，以及总体结果、失败位置和实际指标；失败或输入不匹配时返回非零。

完整优化验收还要求原有分词、内存、生成回归通过。统一入口的
`OPTIMIZATION_EQUIVALENCE_PASS_NUMERICAL_REVIEW_REQUIRED` 表示满足此 V0 等价性
规则及已有回归，**不是**整个模型相对 Transformers 的数学正确性证明。
