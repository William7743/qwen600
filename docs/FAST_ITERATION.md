# V0 的快速迭代和融合验收

> 历史/维护者流程：实习生默认入口已改为 ShareGPT logits 与 benchmark。见 [当前维护说明](MAINTAINER_VALIDATION.md)；下文的旧“默认”指历史版本。

此更新只改变测试工具和流程，不改变 V0 的推理算子、前向逻辑或性能负载。
独立算子测试不能约束学生必须使用某一种实现结构。融合多个算子后，应比较融合部分
的输出或最终 logits，而不要求产生已经消除的中间张量。

## 入口区别

| 命令 | 用途 | 是否要求旧 CUDA 算子 |
| --- | --- | --- |
| `tests/iterate.py check` | 6 条固定历史、55 个全词表 logits 向量 | 否 |
| `tests/iterate.py bench` | 四组短性能负载，1 次预热、每条测 1 次 | 否 |
| `tests/iterate.py diagnose --case NAME` | 指定用例的已有层快照诊断 | 否；需要对应层观察点 |
| `tests/iterate.py full` | 34 条历史、307 个 logits 向量，包含 8192 长度 | 否 |
| `tests/validate.py` | 环境/模型检查、分词、CPU ASan/采样、完整 logits | 否 |
| `tests/validate.py --legacy-operators` | 旧独立算子、逐层与 Transformers 诊断 | 是；仅历史复现/诊断 |

新默认入口不会去查找 `operator_probe` / `correctness_probe`。它只需要 `iteration_probe`
和 CPU 分词/采样探针。模型探针通过前向接口测试真实实现；兼容原 V0 三参数 forward，
以及提供第四个输出控制参数的实现。适配器不会给 V0 添加计算优化。

如果优化实现新增独立 prefill API 或其他执行路径，必须同步修改模型探针和 benchmark
调用的适配代码，不能用保留的旧前向路径替代对真实运行路径的验证。

## 构建（本机已有环境）

在项目根目录执行，其他机器调整路径与 GPU 架构：

```bash
cmake -S . -B build-iteration -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_PREFIX_PATH=/opt/anaconda3 -DQWEN_BUILD_ITERATION=ON -DQWEN_BUILD_BENCHMARKS=ON
cmake --build build-iteration --target iteration_probe benchmark_probe -j 4
```

不要为了新流程启用 `QWEN_BUILD_TESTS`，它包含旧独立 CUDA 算子。CPU ASan 单独构建，
命令及现有库要求见 [LOCAL_VALIDATION.md](LOCAL_VALIDATION.md)。`validate.py` 会确认
CPU 探针实际启用了 ASan，并检查模型哈希和参考依赖版本，不会自动下载或安装。

## 优化前准备参考包

本机已经从可信 V0 结果准备了 `build-iteration-reference`。该目录被 Git 忽略，需由
出题方保存，不可在清理 build 目录时误删。manifest 中的文件哈希用于发现修改，出题方
还应另存 manifest 哈希，防止整个参考包被替换。

新 clone 由出题方在**尚未优化的 V0**上执行一次旧参考测试、冻结，再准备新参考包：

```bash
# 按 tests/README.md 构建旧探针和 CPU ASan 后，显式运行旧参考流程。
python tests/validate.py --legacy-operators --model /path/to/model --build-dir /path/to/legacy-build --asan-build-dir /path/to/asan-build --output /path/to/v0-validation
python tests/check_optimization.py freeze --results /path/to/v0-validation/results --baseline /path/to/frozen-v0
python tests/iterate.py prepare --baseline /path/to/frozen-v0 --long-reference /path/to/v0-validation/results --output build-iteration-reference
```

这一步用于出题方准备可信预期输出，优化后的学生版本不必再编译旧算子。
准备输入来自原 V0 的 33 条加强用例和 8192-token 边界序列；不需要优化参考实现，
也不能让学生拿自己的结果重新冻结。V0 相对 Transformers 的完整数值验收仍待完成。

本机已有结果对应的准备命令（已完成，不要重复覆盖）：

```bash
/home/msganzy/vllm-shared/base-env/bin/python tests/iterate.py prepare --baseline build-v0-reference --long-reference build-release-check/validation-8192/results --output build-iteration-reference
```

## 每轮优化

每次改代码先构建，再查 logits，通过后测短 benchmark：

```bash
cmake --build build-iteration --target iteration_probe benchmark_probe -j 4
/home/msganzy/vllm-shared/base-env/bin/python tests/iterate.py check --output build-iteration/check-001
/home/msganzy/vllm-shared/base-env/bin/python tests/iterate.py bench --output build-iteration/bench-001
```

输出目录必须为空，下一轮换成 `002`。参数 `--model`、`--baseline`、`--build-dir` 可覆盖
本机默认路径。默认模型为 `/home/msganzy/vllm-shared/models/Qwen3-0.6B`，其余默认目录
位于当前仓库根目录下。快速验证不加载 Transformers 模型，只用 NumPy 比较冻结 logits。

六条包含中英文、代码、空白、129 和 1025 边界；所有采样位置比较整个词表，不只是 top-1。
两边使用相同 token ID 和参考历史，避免自由生成的分歧放大掩盖计算误差。
阈值仍是每个位置 MAE ≤0.05、最大误差 ≤0.5、概率 TV ≤0.02、V0 对候选 top-1 的
分数损失 ≤0.125，拒绝 NaN/Inf。快速通过只表示适合继续迭代，不等于完整交付通过。

四组短 benchmark 是 16/16、256/16、1024/16、16/256，分别观察固定开销、prefill 和 decode。
长历史 Attention 改动可加 `--long-decode` 测 1024/256；小幅提升用 `--repeats 3` 或更多
重复次数确认。V0 和候选必须使用相同配置，短集成绩不能和全量成绩直接比较。

## 失败后的诊断

```bash
/home/msganzy/vllm-shared/base-env/bin/python tests/iterate.py diagnose --case en_summary --output build-iteration/diagnose-001
```

先查看失败的采样位置，再检查该用例已有的层快照。阶段 0 为 embedding，1～28 为层输出，
29 为最终归一化。层级诊断会区分首次不完全相同与首次超出旧层容差，两者并不等价。
已有层快照只覆盖 33 条用例的首、末输入位置，不覆盖所有 decode 位置；8192 序列只有
logits 参考。缺少对应位置时，应从可信 V0 补采或缩小复现输入，不能声称已定位所有层。

## 阶段验收

```bash
/home/msganzy/vllm-shared/base-env/bin/python tests/validate.py --model /home/msganzy/vllm-shared/models/Qwen3-0.6B --build-dir build-iteration --asan-build-dir build-release-asan --output build-iteration/validation-001
```

默认状态 `MODEL_LOGITS_AND_CPU_REGRESSIONS_PASS_NUMERICAL_REVIEW_REQUIRED` 明确表示
模型 logits 与 CPU 回归通过；**不包含 GPU 内存检查或自由生成回归**，也不认证完整的
Transformers 数学一致性。完整交付还需检查实际生成流程，并在真实优化路径上运行：

```bash
/home/msganzy/vllm-shared/base-env/bin/python tests/iterate.py full --sanitizer memcheck --output build-iteration/full-memory-001
```

布局/索引改动后可以先用 `check --sanitizer memcheck`；共享内存改动可加 racecheck。
Sanitizer 明显慢于正常检查，不能拿其耗时作为性能成绩。默认超时 7200 秒，可用
`--timeout-seconds` 调整；超时会终止探针进程组。日常不必每轮运行长内存测试。
数值验收后再运行已有 9 组及 100 条负载完成性能比较。

新结构契约见 [acceptance_contract.json](../tests/acceptance_contract.json)。
旧 `optimization_policy.json` 中的独立算子/层约束保留给旧检查器复现历史结果；新流程
只沿用其 logits 数值阈值，不把旧算子存在性作为通过条件。

## 本机验证记录

本轮在没有 `operator_probe` / `correctness_probe` 的新构建目录中运行默认入口，
分词、CPU ASan/采样和全部 307 个 logits 向量通过。普通短检查约 11 秒，完整 logits
约 171 秒，四组短 benchmark 约 16 秒；这些是本机单次耗时，不是优化收益声明。
推理源文件哈希保持不变。报告见 [validation-infrastructure](validation-infrastructure/)。
