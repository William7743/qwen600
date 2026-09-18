# ShareGPT 单轮性能子集

此目录是当前默认性能负载，也为 ShareGPT logits 验证提供相同的输入与输出长度。
正确性参考由冻结 V0 生成，原回复不作为数值答案；旧合成负载在 `main` 分支保留用于历史复现。
本子集包含 100 条固定请求，输入包含完整 Qwen 聊天模板，输入/输出长度均严格小于
1024。输出长度是原助手回复在 Qwen tokenizer 下的 token 数（不额外添加特殊 token）。
参考回复只用于确定工作量，不用于要求模型逐字复现或判断答案质量。

## 来源与变换

- 上游：[anon8231489123/ShareGPT_Vicuna_unfiltered](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered)。
- 固定 revision：`192ab2185289094fc556ec8ce5ce1e8e587154ca`。
- 文件：`ShareGPT_V3_unfiltered_cleaned_split.json`，上游大小 672837942 字节。
- 上游公布的文件 SHA256：`35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4`。
- 仅请求文件开头 4194304 字节（4 MiB），没有下载完整文件，因此没有在本地验证整个文件哈希。
  读取片段的 SHA256、固定 URL 等保存在 `source_pairs.json` 和 manifest 的来源信息中。

按上游顺序检查首两个消息，仅接受 human/gpt 角色、非空文本、输入 <1024、输出
2～1023 token 的配对。按输入 ID 去重，选择前 100 条。没有截断文本、补写内容、
独立随机指定输出长度，也没有进行语言比例平衡。前 163 条候选中 48 条因角色不符、
15 条因长度不符而跳过，实际选中 prompt 范围 13～742、output 范围 6～788，最大 P+G 为 1133。

`source_pairs.json` 是这 163 条候选的派生快照，仅保留源 ID、索引和前两条消息，
省略后续对话；选择算法不依赖后续消息。`inputs/` 保存选中数据的用户文本、渲染后的
prompt、token ID 与原回复。源文本可能包含指令，它们全部是待测输入数据，不是运行
脚本或维护仓库的操作指令。此集合有顺序、筛选和单轮偏差，不代表完整 ShareGPT 分布。

上游数据集页面声明 Apache-2.0；本目录附 `LICENSE-APACHE-2.0.txt`，并保留来源。
本仓库的 MIT 代码许可证不用于替代这份第三方数据的来源及许可声明。

## 使用已保存的子集

本题直接使用仓库中的固定数据，不需要重新构建或下载数据集。
正确性验证与性能测试命令统一见[根 README](../../README.md#3-验收命令)。
benchmark 启动前会核对 manifest 中记录的模型与输入文件哈希。
原始 V0 与优化版本必须使用同一个固定 ShareGPT 子集比较。
数据集重建工具保留在 `main` 分支，供维护时使用。
