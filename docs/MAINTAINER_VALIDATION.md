# 维护者：参考包与保留的回归工具

实习生默认只使用项目 README 的 ShareGPT logits 检查和 benchmark。本题由学生在自己的兼容硬件和环境中、优化前用原始V0生成参考；
维护者在需要时使用旧回归工具定位问题。正式规则见[题目](../PROBLEM.md)。

## 一次性准备 ShareGPT 参考包

在尚未优化的可信 V0 checkout 中，按 README 构建 `build-sharegpt` 后运行：

```bash
python tests/sharegpt_check.py freeze --model /path/to/Qwen3-0.6B
```

此命令拒绝生产源码与冻结 V0 提交 `b5c1869` 不一致的 checkout，并重新构建探针。
它在本地为 100 条输入生成贪心 V0 续写，再用固定历史采集 600 个全词表 logits。
原 ShareGPT 回复只确定 G；历史为 prompt 加 V0 生成的前 G−1 个 token。
每条零基采样位置为 `{0, (P−1)//2, P−1, P, P+(G−2)//2, P+G−2}`。
模型只使用本地文件，不自动下载。命令拒绝覆盖非空参考目录。

生成 `build-sharegpt-reference/manifest.json`、`logits.f32` 和 `histories/`；
学生应保存这三个项目及生成日志；`generation/` 的时间不作为性能基线，另行使用一次预热的benchmark建立基线。
完整 logits 数组为 600 × 151936 × 4 = 364646400 字节（约 347.8 MiB）。
`tests/sharegpt_reference.json` 初始未设置manifest哈希，`freeze` 自动写入学生本机生成的哈希；
学生将其记录在过程文档中，并在后续优化中保持参考与哈希不变。
检查时同时校验参考 manifest、模型、历史、数组及误差阈值，拒绝被修改的参考包。
不同学生的GPU和工具链可以不同，参考哈希也可能不同。若本人更换环境，应在未优化V0中重新建立参考和性能基线，再在相同新环境重测候选。

`--quick` 固定取按 P+G 排序后下标 0、19、39、59、79、99 的 6 条，名单写入 manifest，
不能因候选失败临时改选。全量验收始终运行 100 条；失败后报告具体用例、位置与误差。

## 保留的工具

| 工具 | 维护用途 |
| --- | --- |
| `tests/iterate.py full` | 旧 34 条固定历史 / 307 个位置，包含 8192 长历史 |
| `tests/iterate.py diagnose` | 旧用例已有的层输出诊断；不直接接受 ShareGPT 用例 ID |
| `tests/validate.py` | 旧模型/CPU 回归流程 |
| `tests/validate.py --legacy-operators` | 原独立算子、逐层及外部参考诊断 |
| `benchmarks/run_benchmark.py --dataset fixed9` / `diverse100` | 原性能负载复现 |

旧命令依赖原有参考包和构建，完整说明保留在 [旧快速迭代文档](FAST_ITERATION.md) 与
[旧正确性文档](../tests/README.md)。它们描述的是维护者流程，不是当前实习生必经步骤。
旧参考包和原报告不能覆盖或当作可删除缓存。

ShareGPT 检查不涵盖分词特殊字符、随机采样、内存越界或全部上下文长度。
布局/索引/同步改动出现问题时，维护者可追加真实前向的 sanitizer 检查；
融合后只检查仍有意义的接口，不能强制恢复已消除的独立算子。
外部 Transformers 数值审查仍与 V0 相对等价性分开。
