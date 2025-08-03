# GPU内存带宽测试工具（Triad）详解 / GPU Memory Bandwidth Test Tool (Triad)

## 1. 工具概述 / Tool Overview

`triad` 是基于经典STREAM基准测试的GPU内存带宽测量工具。该工具实现了Triad操作（a[i] = b[i] + alpha * c[i]），用于评估GPU的实际内存带宽性能，并可用于研究多GPU环境下的内存带宽竞争问题。

## 2. 技术原理 / Technical Principles

### 2.1 STREAM Triad基准测试 / STREAM Triad Benchmark

Triad操作是STREAM基准测试中最复杂的操作，涉及：

- 2次内存读取（数组b和c）
- 1次内存写入（数组a）
- 1次浮点乘法和1次浮点加法

```cuda
// 核心计算
a[i] = b[i] + alpha * c[i]

// 内存访问模式
Read:  b[i], c[i]    // 2 * sizeof(float) * N
Write: a[i]          // 1 * sizeof(float) * N
Total: 3 * sizeof(float) * N bytes
```

### 2.2 带宽计算公式 / Bandwidth Calculation

```
有效带宽 = (传输的字节数) / (执行时间)
        = (3 * N * sizeof(float)) / (time_in_seconds)
        = (3 * N * 4) / (time_ms * 1e-3) / (1 << 30) GB/s
```

## 3. 实现架构 / Implementation Architecture

### 3.1 代码结构 / Code Structure

```cpp
主要组件：
├── 信号处理
│   ├── g_running (原子标志)
│   └── signal_handler() (SIGINT/SIGTERM处理)
├── CUDA核函数
│   └── triad() (网格跨步循环实现)
└── 主程序
    ├── 参数解析
    ├── GPU内存分配
    ├── 性能测试循环
    └── 资源清理
```

### 3.2 核函数优化 / Kernel Optimization

```cuda
__global__ void triad(float* __restrict__ a,
                      const float* __restrict__ b,
                      const float* __restrict__ c,
                      float alpha,
                      size_t n) {
    // 网格跨步循环（Grid-stride loop）
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
  
    for (size_t i = tid; i < n; i += stride) {
        a[i] = b[i] + alpha * c[i];
    }
}
```

**优化技术**：

- `__restrict__` 关键字：告诉编译器指针不会重叠
- 网格跨步循环：处理任意大小的数据
- 合并内存访问：连续的内存访问模式

## 4. 功能特性 / Features

### 4.1 核心功能 / Core Features

1. **高精度带宽测量**

   - 使用CUDA Event进行精确计时
   - 实时计算和显示有效带宽
2. **灵活的参数配置**

   - 可配置数组大小（元素数量）
   - 可配置迭代次数
   - 支持命令行参数

### 4.2 输出格式 / Output Format

```
Iter   0 | 12.456 ms | Effective BW 325.67 GB/s
Iter   1 | 12.234 ms | Effective BW 331.59 GB/s
...
```

每行包含：

- 迭代编号
- 执行时间（毫秒）
- 有效带宽（GB/s）

## 5. 编译

`build.sh` 关键配置：

```bash
ARCH=sm_90              # GPU架构（根据实际GPU调整）
OPT="-O3"              # 最高优化级别
DEBUG="-lineinfo"       # 保留调试信息

nvcc $SRC -o $OUT \
     $OPT $DEBUG \
     -std=c++17         # C++17标准
     -arch=$ARCH        # 目标GPU架构
```

## 6. 使用方法 / Usage

### 6.1 基本用法 / Basic Usage

```bash
# 默认运行（256M元素，100次迭代）
./triad

# 自定义参数
./triad [elements] [iterations]

# 示例：1G元素，1000次迭代
./triad 1073741824 1000
```

### 6.2 运行脚本功能 / Run Script Features

`run.sh` 提供了高级运行选项：

```bash
# 基本运行
./run.sh              # 在GPU 0上运行

# 指定GPU
./run.sh 1            # 在GPU 1上运行

# 性能分析模式
./run.sh 0 ncu        # 使用Nsight Compute分析
```

## 7. 性能分析 / Performance Analysis

### 7.1 理论带宽计算 / Theoretical Bandwidth

```
GPU理论带宽 = 内存频率 × 总线宽度 × 2（DDR）

示例：
- V100:  877 MHz × 4096-bit × 2 = 900 GB/s
- A100: 1215 MHz × 5120-bit × 2 = 1555 GB/s  
- H100: 1593 MHz × 5120-bit × 2 = 3.35 TB/s
```
