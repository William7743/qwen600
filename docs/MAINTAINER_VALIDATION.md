# 本机 V0 参考包说明

学生在自己的兼容硬件和环境中、优化前用原始 V0 生成参考。
构建配置、运行命令与验收规则统一见[项目 README](../README.md)。

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

## 检查范围

本题目验收完整模型的最终 logits，不要求独立算子或逐层输出检查。
ShareGPT 检查不覆盖分词特殊字符、随机采样、全部上下文边界或全部内存安全情形。
旧回归工具和历史结果保留在 `main` 分支，需要诊断时可查阅。
