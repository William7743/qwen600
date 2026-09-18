# ShareGPT 单请求 benchmark

默认使用已保存的 `sharegpt100`，100 条固定单轮请求，batch=1、并发请求数=1。
输入 13～742 token，输出 6～788 token；两者各自均小于 1024，实际最大 P+G 为 1133。
来源、筛选和许可证见 [子集说明](sharegpt100/README.md)。这是固定的前缀筛选子集，
不代表完整 ShareGPT 分布，也不覆盖全部 2048 长度范围。

模型加载一次，默认先做一次真实请求预热，再每条正式测一次；每条重置逻辑历史。
贪心生成，忽略 EOS，严格完成指定 G 个 token。预热固定取 s038（P=432、G=362），
不计入汇总；即使只选择部分请求，预热用例不变。所有输入在计时前读取，不重新分词。

## 构建与运行

仓库根目录执行，工具链、模型与依赖均使用本地已有文件：

```bash
cmake -S . -B build-sharegpt -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_PREFIX_PATH=/opt/anaconda3 -DQWEN_BUILD_ITERATION=ON -DQWEN_BUILD_BENCHMARKS=ON
cmake --build build-sharegpt --target iteration_probe benchmark_probe -j 4
python benchmarks/run_benchmark.py --model /path/to/Qwen3-0.6B --output build-sharegpt/bench-001
```

默认构建目录为 `build-sharegpt`，可用 `--build-dir` 指定。结果目录必须为空。
`--quick` 选与 logits 快速检查相同的 6 条；`--case s001` 选单条，`--repeats 3` 重复测量，`--warmup 0` 仅供诊断。
不同预热、重复次数或用例选择的结果不能混作同一组成绩。
成功退出只表示计时完成；正确性由 `tests/sharegpt_check.py check` 单独判定。

## 指标定义

设第 i 条请求输入 P_i 个 token，输出 G_i 个 token。
请求开始于输入就绪且此前 GPU 工作同步完成后；总耗时结束于最后 token ID 在 CPU 可用时；最终错误检查和兜底同步在计时之外。
记总耗时 T_i，decode 耗时 D_i，以下吞吐公式的时间单位为秒。

| 类别 | 指标 / 字段 | 定义 |
| --- | --- | --- |
| 延迟 | Prefill / `prefill_ms` | 处理全部 prompt，到末位置 logits 在主机可用 |
| 延迟 | TTFT / `ttft_ms` | 请求开始到首个输出 token ID 可用，含首次采样 |
| 延迟 | Decode / `decode_ms` | 首个 token 可用到请求结束，含后续 G−1 步，不含末尾兜底同步检查 |
| 延迟 | TPOT / `tpot_ms` | D_i / (G_i−1) |
| 延迟 | ITL / `itl_ms` | 相邻输出 token ID 在主机可用的时间差，每请求保存 G_i−1 个值 |
| 延迟 | 请求总耗时 / `total_ms` | TTFT + decode 耗时 |
| 吞吐 | 输入与输出合计 / `total_tokens_per_second` | ∑(P_i+G_i) / ∑T_i |
| 吞吐 | 全请求输出 / `output_tokens_per_second` | ∑G_i / ∑T_i |
| 吞吐 | Decode / `overall_decode_tokens_per_second` | ∑(G_i−1) / ∑D_i |
| 吞吐 | 串行请求 / `serial_requests_per_second` | 请求数 / ∑T_i，单位 request/s |
| 资源 | 显存观测峰值 / `memory.observed_peak_mib` | 独立 `--memory-only` 运行中 nvidia-smi 的进程显存观测峰值 |

首 token 来自最后一次 prompt 前向，后续只有 G−1 次 decode 前向。
ITL 是实际逐 token 时间戳之差，不能用重复的 TPOT 代替。protocol 4 的
`sum(itl_ms) = decode_ms`，单个请求的 ITL 均值等于 TPOT；跨请求汇总的权重不同，均值仍可不同。
兼容字段 `decode_tail_ms` 固定为 0。没有网络与响应传输层。所有记录缓冲区在整轮前分配，
统计、JSON 输出及文件写入在全部推理结束后进行；ITL 打点本身的开销仍计入时间。

请求计时包含推理内部数据传输与采样，排除模型加载、分词、文件读写、打印和请求间空隙。
本项目吞吐使用各请求的引擎耗时之和，**不是整个脚本的墙钟运行时间**。
请求吞吐表示 batch=1、并发=1 时当前固定负载的串行处理速度，不能当作多并发服务容量。
V0 与候选必须使用相同负载和计时协议。当前为 protocol 4；请求结束点和监控方式已改变，
不能与 protocol 2/3 的旧表混作严格对照，需重测双方。

## 分位数与结果文件

Prefill、TTFT、decode、TPOT、请求总耗时各自按正式请求统计 mean、P50、P95、P99。
ITL 则汇集所有正式请求的实际 token 间隔再统计，长输出请求贡献更多间隔。
P50 为中位数；P95/P99 用于观察较慢的样本。分位数采用排序后 `(n−1)*p` 位置的线性插值。
一个样本时三个分位数相同。100 条一次测量的 P99 只描述此样本，不能证明线上尾延迟或重复运行稳定性。
按固定用例重复测量时，`summary.json` 另保存各用例的分布。

- `raw.jsonl`：预热与正式请求的原始标量、逐 token ITL、完整生成 ID。
- `results.csv`：正式请求的 Prefill、TTFT、TPOT、decode、decode token/s、总耗时与 ITL 均值。
- `itl.csv`：每个正式输出间隔；`output_token_index` 为零基，1 表示第 1 到第 2 个输出 token 的间隔。
- `aggregate.json`：延迟均值与分位数、所有吞吐、总 token 数与计时分母。
- `summary.json`：汇总、每用例分布、模型加载时间、构建/版本/设备信息与显存观测。
- `stderr.log`、`plan.txt`：引擎日志和执行计划；`memory.jsonl` 仅在独立显存运行中生成。

默认延迟测试不启动显存轮询，显存字段为 null。使用独立 `--memory-only` 运行时约每秒查询一次，
范围包含加载、预热及请求，可能漏掉瞬时峰值，不声称精确峰值。资源运行的时间只用于诊断。
无法取得进程显存数据时填 null 并附原因。吞吐按计数总和除以时间总和，不能平均每条 token/s。

## 显存与计时开销验证

```bash
# 与正式延迟测量分开运行；建议使用相同完整负载。
python benchmarks/run_benchmark.py --model /path/to/Qwen3-0.6B --output build-sharegpt/memory-001 --memory-only
# 同一可执行文件，固定六条 ShareGPT，ABBA+BAAB，各进程每条三次。
python benchmarks/check_timing_overhead.py --model /path/to/Qwen3-0.6B --build-dir build-sharegpt --output build-sharegpt/timing-check-001
```

`--no-itl` 仅供对照校准，保留 Prefill/TTFT/请求结束时间，但不记录中间 token 时间戳。
其 ITL 字段为 null，不输出虚构的 ITL 分位数；正式报告继续使用默认完整打点模式。
校准结论与限制见 [计时说明](../docs/TIMING_PROTOCOL.md)。

## 历史负载

`--dataset fixed9` 和 `--dataset diverse100` 仍可运行；文件与原 manifest 保留。
前者是 P/G 分别取 16、256、1024 的九组组合；后者是 100 条固定合成文本。
它们不属于新的默认验收。旧集合生成脚本继续保留，不运行新下载、不重新抽样。
