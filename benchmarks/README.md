# V0/V1 单请求性能负载

这是固定工作量的性能输入集，不是数值正确性测试集，也不包含标准答案。
计时入口为 `run_benchmark.py`，调用独立的 `benchmark_probe`，直接复用生产
`forward()` 和贪心采样，不修改生产推理路径。生成器与计时器分离，计时器仅依赖
Python 标准库和本机已构建的 CUDA 程序，无需安装 PyTorch 或下载软件。
V0 的正式性能基线需要在数值正确性验收后测量；此时不把当前提交标记为 V0。

## 构建与运行

在仓库根目录构建本机 RTX 3090 版本（其他机器调整架构和依赖位置）：

```bash
cmake -S . -B build-benchmark -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_PREFIX_PATH=/opt/anaconda3 -DQWEN_BUILD_BENCHMARKS=ON
cmake --build build-benchmark --target benchmark_probe -j 4
```

一行运行全部九组，每组预热 2 次、测量 5 次：

```bash
/home/msganzy/vllm-shared/base-env/bin/python benchmarks/run_benchmark.py --model /home/msganzy/vllm-shared/models/Qwen3-0.6B --output build-benchmark/results
```

`--build-dir` 默认是仓库的 `build-benchmark`。结果目录必须为空，避免覆盖历史成绩。
只测试一组可添加 `--case p16_g16`；多组选项可重复。`--warmup`、`--repeats` 可调整，
但非默认次数只能作为调试结果，不能混入默认方案成绩。退出码 0 表示计时工作完成，
并不表示数值正确性通过；引擎失败或结果不完整时非零退出并记录失败状态。

终端逐次打印 Prefill、TTFT、TPOT、decode tokens/s、总耗时，最后打印中位数摘要。
输出文件：

- `raw.jsonl`：所有预热及正式请求、逐次耗时、完整输出 token ID。
- `results.csv`：正式请求的逐次指标（不含预热）。
- `summary.json`：每组中位数/最小值/最大值、重复生成是否一致、环境及版本信息。
- `memory.jsonl`：本次引擎进程的显存采样；`stderr.log`：CUDA/引擎错误。
- `plan.txt`：实际执行的固定工作量。

显存当前记录整次进程运行（含加载、预热）的观测峰值，采样间隔为 1 秒加查询开销，
不冒充单个请求的精确峰值。未暂停在加载完成边界采样，因此不提供独立的加载后
进程显存数值。`nvidia-smi` 不可用或无法获得该进程的数据时，峰值为 null 并附原因。

## 数据与复现

- `source.txt`：本项目自编英文工程笔记，无需下载外部数据。
- `inputs/p{16,256,1024}.user.txt`：实际 user 内容；按 token 前缀裁剪，可能在句中结束。
- `inputs/p{16,256,1024}.prompt.txt`：完整聊天模板文本，关闭 thinking。
- `inputs/p{16,256,1024}.ids`：真正送入模型的输入，每行一个十进制 token ID。
- `manifest.json`：模型版本、输入哈希、分词依赖及九组配置。所有文件路径相对本目录。

输入长度包含聊天模板；只裁剪 user 内容，不截断模板或 generation prompt。
运行 benchmark 时直接读取 ID 文件，不重新套模板，也不按字符数推断 token 数。
三种输入是同一来源的前缀，不代表不同领域或所有自然文本的性能。

在仓库根目录用已有环境重建：

```bash
/home/msganzy/vllm-shared/base-env/bin/python benchmarks/build_dataset.py --model /home/msganzy/vllm-shared/models/Qwen3-0.6B
```

不修改文件、重新生成并逐字节核验：

```bash
/home/msganzy/vllm-shared/base-env/bin/python benchmarks/build_dataset.py --model /home/msganzy/vllm-shared/models/Qwen3-0.6B --check
```

其他机器替换 Python 和模型路径即可。生成器只读取本地 tokenizer，检查文件哈希和
分词相关依赖版本，不下载依赖或权重。manifest 记录权重哈希，但生成器不读取权重；
未来计时入口仍需核验实际模型权重。

## 九组工作量

| Prompt tokens | Output tokens |
| ---: | --- |
| 16 | 16、256、1024 |
| 256 | 16、256、1024 |
| 1024 | 16、256、1024 |

全部为单 GPU、单实例、batch=1、单请求，无并发。模型加载一次，每组先完整预热
2 次，再正式运行 5 次；每次重置位置与序列状态，不复用上次请求的逻辑历史。
使用贪心采样，忽略 EOS，输出严格达到 G 个 token；不要求 V0/V1 强行使用相同
输出，但正确性需要独立验收。计时期间禁止输出文本或运行 profiler/sanitizer。
G 个输出中，第一个来自最后一次 prompt 前向，因此只有 G−1 次 decode 前向。
最大 P+G 为 2048，不覆盖 8192 上限性能。输入集固定后不能因某项优化获益少而换数据。

## 测量指标与边界

固定工作量计时不包含读取 ID 文件、分词、聊天模板构造、文本解码和终端打印。
使用同步后的 CPU 单调墙钟；当前 forward 包含 host logits 传输，应计入请求开销。
不要把 kernel 提交时间冒充 GPU 执行时间，也不要为了计时给每个 kernel 加同步。

定义 t0 为输入已就绪、此前 GPU 工作同步完成后开始请求的时刻；t1 为全部 P 个
输入前向结束且最后一份 logits 在主机可用的时刻；t2 为完成第一次 argmax、首个
token ID 可用的时刻；t3 为完成后续 G−1 次前向及 argmax、最后一个 token ID 可用
并完成必要同步的时刻。以下所有时间均用毫秒保存。

| 字段 | 计算或定义 |
| --- | --- |
| `prefill_ms` | t1−t0；当前实现逐 token 处理 prompt，整个循环计入 |
| `ttft_ms` | t2−t0；本项是预分词输入的引擎首 token 延迟，不是服务端到端延迟 |
| `decode_ms` | t3−t2；包括后续前向、必要数据传输及采样 |
| `tpot_ms` | decode_ms / (G−1)，单次请求的平均值；即原规范的 decode_ms_per_token |
| `decode_tokens_per_second` | 1000 × (G−1) / decode_ms |
| `total_ms` | t3−t0，应等于 ttft_ms + decode_ms |
| `model_load_ms` | CUDA 上下文初始化完成后，加载权重及分配状态到同步完成；单独记录 |
| `memory.observed_peak_mib` | 整次运行的进程设备内存采样最大值，MiB=2^20 字节；方法与范围记录在 memory 中 |

每组保存 5 次原始记录，再分别报告指标的中位数、最小值和最大值。吞吐先按每次
测量计算，再汇总。显存峰值如通过采样得到，应标为“观测峰值”，可能漏掉短暂分配；
工具无法获得进程级数据时填 null 和原因，不能用其他进程也占用的设备总量替代。
当前 KV cache 预分配至 SEQ_LEN，短输入的显存占用不一定更低。

同时记录代码 commit/dirty 状态、模型/输入哈希、构建类型、GPU 架构参数、编译器、
CUDA/驱动、GPU 型号、温度/频率/功耗状态。禁止其他任务争用 GPU；正式成绩使用
Release 构建。V0/V1 使用相同配置与测量边界。每组分别展示加速或退化，不只报最好项。
