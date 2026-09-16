# 本机复现命令与验证边界

这些路径仅记录维护者机器的现有依赖，不是仓库代码或通用测试入口的要求。
通用流程见 [tests/README.md](../tests/README.md)。

在任意新的本仓库 clone 根目录执行：

```bash
cmake -S . -B build-check \
  -DCMAKE_PREFIX_PATH=/opt/anaconda3 \
  -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release \
  -DQWEN_BUILD_TESTS=ON -DQWEN_ENABLE_ASAN=OFF
cmake --build build-check -j 4

cmake -S tests -B build-asan \
  -DCMAKE_PREFIX_PATH=/opt/anaconda3 \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
  -DCMAKE_CXX_FLAGS='-I/usr/include/c++/4.8.5/x86_64-redhat-linux -L/opt/rh/devtoolset-8/root/usr/lib/gcc/x86_64-redhat-linux/8' \
  -DQWEN_ENABLE_ASAN=ON
cmake --build build-asan -j 4

/home/msganzy/vllm-shared/base-env/bin/python tests/validate.py \
  --model /home/msganzy/vllm-shared/models/Qwen3-0.6B \
  --build-dir build-check --asan-build-dir build-asan \
  --output build-check/validation
```

此处 GCC 8.3.1 可作为 nvcc host compiler，但缺少对应的 ASan runtime，
所以 CPU 检查使用 Clang 3.4.2。通用环境通常只需要一个支持 ASan 的现代编译器。

验证结论应限定为：干净的 Git checkout、重新编译、重新导出分词器，
在本机现有工具链和 Python 参考依赖下运行。它不是在另一台 GPU 或全新操作系统上
重新安装所有依赖的证明；依赖版本与来源记录在 `tests/reference.json`。
