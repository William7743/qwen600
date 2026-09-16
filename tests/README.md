# 从全新 clone 复现正确性回归

本流程对照本项目与 Hugging Face Transformers，限定输入与生成累计不超过
1024 token。它验证分词、内存边界、采样和指定生成用例；**前向数值差异的完整
验收仍未完成**。退出码 0 不代表模型的所有数学计算已经认证正确。

## 1. 环境与版本

需要 Linux、支持 BF16 的 NVIDIA Ampere 或更新 GPU、可用 CUDA Toolkit/nvcc、
兼容驱动、C++17 编译器、CMake >=3.20、PCRE2 8-bit、ICU uc 和 CPU AddressSanitizer。
本次基线使用 RTX 3090 24GB、CUDA Toolkit 12.4.131、驱动 550.54.15、PCRE2 10.42、
ICU 73.1、Python 3.10。CUDA Toolkit 与 PyTorch 自带的 CUDA runtime 是不同依赖。

Python 主要依赖版本保存在 [requirements-reference.txt](requirements-reference.txt)，
完整的基线元数据及模型哈希保存在 [reference.json](reference.json)。这不是完整的
操作系统镜像或全部传递依赖锁文件，不能保证任意操作系统/GPU 上得到逐位相同数值。

优先复用已有 Python 环境和本地模型：直接使用该环境的 Python 执行第 4 节命令，
验证入口会检查依赖版本，缺失或不符时会报告错误，不会自动安装或下载。
本机已有环境的完整命令见 [LOCAL_VALIDATION.md](../docs/LOCAL_VALIDATION.md)。

仅在没有可用环境、且需要自行安装时，在新 Python 3.10 虚拟环境中准备参考依赖
（以下安装命令会下载软件包）：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'torch==2.11.0+cu129' --index-url https://download.pytorch.org/whl/cu129
python -m pip install -r tests/requirements-reference.txt
```

[PyTorch 官方 wheel 索引](https://download.pytorch.org/whl/cu129/torch/)中提供该版本。
Linux wheel 使用 manylinux_2_28，不能把本机经过配置的环境等同于原生 CentOS 7。
PCRE2/ICU 的开发头文件和库需要另行安装；非系统安装位置通过
`-DCMAKE_PREFIX_PATH=/your/dependency/prefix` 指定，代码不依赖 `/opt/anaconda3`。

验证入口默认要求主要 Python 依赖及 Unicode 库与基线版本相符；如要探索不同版本，
显式添加 `--allow-version-drift`，报告会标记 `NON_REFERENCE`，不会冒充参考环境复现。

## 2. 模型文件

从仓库根目录设置自己的模型位置。已有同版模型可以直接复用，无需再次下载：

```bash
export QWEN_MODEL_DIR=/absolute/path/to/Qwen3-0.6B
```

没有模型时，可按锁定 revision 下载验证所需的五个源文件：

```bash
python - <<'PY'
import json, os
from huggingface_hub import snapshot_download
lock = json.load(open('tests/reference.json'))['model']
snapshot_download(repo_id=lock['repo_id'], revision=lock['revision'],
                  allow_patterns=list(lock['sha256']), local_dir=os.environ['QWEN_MODEL_DIR'])
PY
```

模型为 `Qwen/Qwen3-0.6B`，revision 为
`c1899de289a04d12100db370d81485cdf75e47ca`。权重 SHA256 为
`f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`。
验证入口会检查全部五个源文件哈希，不匹配就拒绝运行，因为权重加载器依赖固定布局。

## 3. 构建

以下示例的 `86` 对应 RTX 3090；其他 GPU 需要选择其架构，并使用支持该架构的 Toolkit。

```bash
cmake -S . -B build-check \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DQWEN_BUILD_TESTS=ON -DQWEN_ENABLE_ASAN=ON
cmake --build build-check -j 4
```

如依赖不在系统路径，在配置命令中追加 `-DCMAKE_PREFIX_PATH=...`。
如 nvcc 需要特定 host compiler，可追加 `-DCMAKE_CUDA_HOST_COMPILER=/path/to/g++`。
构建产物包括 `qwen600` 与 `bin/{correctness_probe,tokenizer_probe,edge_probe}`。
ASan 仅作用于 CPU 测试程序，不给 CUDA 模型探针添加 sanitizer。

## 4. 一条命令运行回归

```bash
python tests/validate.py \
  --model "$QWEN_MODEL_DIR" \
  --build-dir build-check \
  --output build-check/validation
```

这个入口：

1. 检查 Python/Unicode 库版本、CUDA 可用性、模型 SHA256 和 ASan 编译标识。
2. 在输出目录建立模型源文件的符号链接，重新导出 QTK2 分词器和模板。
   不修改源模型目录，也不复用其中已有的 tokenizer.bin。
3. 运行扩展分词、ASan/采样和模型对照，保存每一步日志及结果。

导出及推理使用本地文件，禁用 Hugging Face 在线访问。只在上面的下载步骤需要网络。
如果要手动启动本次构建的 CLI，可使用输出中的模型目录：

```bash
./build-check/qwen600 build-check/validation/model -r 0 -t 0 \
  -i 'What is the capital of France? Answer in one short sentence.'
```

## 5. 如何判断结果

| 检查 | 通过标准 |
| --- | --- |
| 模板 | 4/4 字符串完全一致 |
| 原有分词 | 12/12 token ID 序列完全一致 |
| 扩展分词 | 842/842 token ID 序列完全一致（包含 NFC、空白、代码、特殊 token、NUL） |
| Attention 覆盖 | 全 1 Q/K 的所有已测有效分数等于 sqrt(128)，绝对误差 <=1e-4 |
| `<`、top-k 边界 | 实际启用 ASan，测试正常退出且没有内存诊断 |
| 等概率采样 | 2000 次，单候选频率在 40%~60%，固定种子 42 |
| nucleus 阈值 | 两个等概率候选在 top-p=0.5 时只保留一个 |
| 三组生成及 teacher forcing | 与参考 token 序列/逐步 top-1 完全一致 |
| logits | 必须全部有限；MAE/最大误差/余弦/概率 TV 仅作为诊断，数值容差尚未验收 |

`summary.json` 是总报告：

- 退出码 **0**：`REGRESSION_CHECKS_PASS_NUMERICAL_REVIEW_REQUIRED`，约定回归检查通过，数值验收待定。
- 退出码 **1**：至少一项回归失败，查看 `*.log` 与 `results/*.json`。
- 退出码 **2**：依赖、模型、ASan 或执行环境有问题，查看总报告的 `error`。

不能仅凭生成文本相同、logits 高余弦相似度，或退出码 0 宣称全面正确。
本验证不认证生产 CLI 的多轮聊天，也不改变其 `SEQ_LEN=8192` 配置；超过 1024 的已知问题仍在。
详细历史及修复后结果见 [CORRECTNESS.md](../docs/CORRECTNESS.md)。

## 独立 ASan 编译器（旧开发环境）

如果 nvcc 的 host compiler 没有 ASan runtime，可对主构建指定
`-DQWEN_ENABLE_ASAN=OFF`，然后单独配置 CPU 测试：

```bash
cmake -S tests -B build-asan \
  -DCMAKE_CXX_COMPILER=/path/to/asan-capable/clang++ \
  -DQWEN_ENABLE_ASAN=ON
cmake --build build-asan -j 4
python tests/validate.py --model "$QWEN_MODEL_DIR" \
  --build-dir build-check --asan-build-dir build-asan --output build-check/validation
```

省略真正的 ASan 构建会被验证入口拒绝。本机特殊工具链的命令记录在
[LOCAL_VALIDATION.md](../docs/LOCAL_VALIDATION.md)，其他机器不应复制其中的绝对路径。
