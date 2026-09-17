# ShareGPT 默认流程验证

本轮只更新测试工具和默认流程，生产推理源码与数值阈值没有改动。

V0 已冻结 100 条 / 600 个参考向量，并通过 6 条 / 36 个位置的默认快速入口自检。

完整 benchmark 使用本地模型、RTX 3090、Release、CUDA 12.4、SM86，
每版一次真实请求预热，100 条各测一次。ITL 共 24942 个，预热不进入统计。
独立使用 NumPy 核对了六类延迟的 P50/P95/P99，核对 token 总数、各项吞吐分母与 ITL CSV 行数。

此轮用于验证工具输出；没有锁定频率或重复交替测量，不能据此宣称稳定的优化收益。
V0/V1 依次运行，没有并行 GPU 推理；过程中有文档整理及 CPU 侧工具检查。

- [verification.json](verification.json)：覆盖、来源哈希与审计结果。
- [aggregate.json](aggregate.json)：本仓库这次 100 请求的完整指标。
- [results.csv](results.csv)：逐请求结果。
- 本地完整原始记录：`build-sharegpt/bench-full-001/`。
- 本地正确性参考：`build-sharegpt-reference/`，不随 Git 分发。

默认工作方式与命令见 [项目 README](../../README.md)。
