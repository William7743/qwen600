cmake_minimum_required(VERSION 3.20)

# Default to the validated RTX 3090 target; callers can select their own GPU.
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures to build")
endif()
project(QWEN3_600M LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
find_package(CUDAToolkit REQUIRED)
include(cmake/TokenizerDependencies.cmake)

add_executable(qwen600 engine/main.cu)
target_include_directories(qwen600 PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(qwen600 PRIVATE CUDA::cublas qwen_tokenizer_dependencies)
target_compile_options(qwen600 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

option(QWEN_BUILD_TESTS "Build the correctness probes" OFF)
option(QWEN_BUILD_BENCHMARKS "Build the single-request benchmark probe" OFF)
if (QWEN_BUILD_BENCHMARKS)
    add_executable(benchmark_probe benchmarks/benchmark_probe.cu)
    target_include_directories(benchmark_probe PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(benchmark_probe PRIVATE CUDA::cublas)
    target_compile_options(benchmark_probe PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
    set_target_properties(benchmark_probe PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endif()
if (QWEN_BUILD_TESTS)
    add_subdirectory(tests)
endif()

# Build independently of legacy per-kernel probes, including for fused implementations.
option(QWEN_BUILD_ITERATION "Build model-boundary iteration probe" OFF)
if (QWEN_BUILD_ITERATION)
    add_executable(iteration_probe tests/iteration_probe.cu)
    target_include_directories(iteration_probe PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(iteration_probe PRIVATE CUDA::cublas)
    target_compile_options(iteration_probe PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
    set_target_properties(iteration_probe PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endif()
