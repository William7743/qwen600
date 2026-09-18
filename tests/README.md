# 正确性验收工具

环境配置、参考生成与验收命令统一见[根 README](../README.md)。
本目录只保留当前 ShareGPT 模型输出验收与指标计算检查。

| 文件 | 用途 |
| --- | --- |
| `sharegpt_check.py` | 在原始 V0 上生成本机参考，或检查候选完整 logits |
| `iteration_probe.cu` | 调用模型前向并导出指定位置的全词表输出 |
| `logit_metrics.py` | MAE、最大绝对误差、概率TV、top-1分数损失 |
| `optimization_policy.json` | 固定数值阈值 |
| `acceptance_contract.json` | 完整模型输出与protocol 4性能验收范围 |
| `sharegpt_reference.json` | 学生首次生成参考后写入的本机manifest哈希 |
| `reference.json` | 模型名称、版本与哈希；其中旧环境记录仅供参考 |
| `test_benchmark_metrics.py` | 无需GPU的统计公式与计时兼容性检查 |

统计工具的自检可在仓库根目录运行：

```bash
"$QWEN_PYTHON" tests/test_benchmark_metrics.py
```

模型源码、测试输入和四项数值阈值保持原样。旧独立算子、分词/采样及外部参考回归工具保留在 `main` 分支。
