# NCCL Send/Recv 延迟测试工具详解 / NCCL Send/Recv Latency Test Tool

## 1. 工具概述 / Tool Overview

`simple_sendrecv_latency` 是一个专门设计用于精确测量GPU间NCCL点对点通信延迟的工具。该工具提供了微秒级的延迟测量精度，并支持详细的统计分析，包括P50、P90、P95、P99等百分位数指标。

## 2. 技术架构 / Technical Architecture

### 2.1 核心组件 / Core Components

```cpp
主要类和函数:
├── Timer                    // 高精度计时器类
├── calculate_percentile()   // 百分位数计算
├── print_stats()           // 统计数据输出
└── main()                  // 主程序逻辑
    ├── 参数解析
    ├── NCCL初始化
    ├── 性能测试循环
    └── 资源清理
```

### 2.2 测量原理 / Measurement Principle

工具使用以下方法实现高精度延迟测量：

1. **cudaStreamQuery轮询**

   ```cpp
   // 启动NCCL操作
   ncclSend(sendbuff, count, ncclFloat, 1, comm, stream);

   // 轮询等待完成
   cudaError_t err = cudaErrorNotReady;
   while (err == cudaErrorNotReady) {
       err = cudaStreamQuery(stream);
   }
   ```
2. **高精度时钟**

   ```cpp
   class Timer {
       using clock = std::chrono::high_resolution_clock;
       // 提供微秒级精度
   };
   ```

## 3. 功能特性 / Features

### 3.1 支持的通信模式 / Supported Communication Modes

- **点对点通信**: Rank 0 → Rank 1 的单向Send/Recv
- **同步测量**: 每次通信完成后才开始下一次
- **批量测试**: 支持不同消息大小的自动化测试

### 3.2 统计分析功能 / Statistical Analysis

```
统计指标：
- Min/Max: 最小/最大延迟
- P50: 中位数延迟（典型值）
- P90/P95: 90%/95%的请求延迟低于此值
- P99/P99.9: 尾延迟指标
- Average: 平均值
- StdDev: 标准差（稳定性指标）
```

## 4. 编译和部署 / Build and Deployment

### 4.1 环境要求 / Environment Requirements

```bash
# 必需组件
- CUDA >= 12.2
- NCCL >= 2.25
```

### 4.2 编译配置 / Build Configuration

`build.sh` 脚本自动处理编译配置：

```bash
# 关键编译选项
CFLAGS="-O3 -std=c++11"                    # 优化级别和C++标准
INCLUDES="-I$CUDA_HOME/include -I$NCCL_HOME/include"
LIBS="-lnccl -lcudart"                     # 链接NCCL和CUDA运行时

# MPI支持（自动检测）
if command -v mpicc &>/dev/null; then
    CFLAGS+=" -DMPI_SUPPORT"
    LIBS+=" -lmpi"
fi
```

### 4.3 环境变量配置 / Environment Variables

```bash
# 路径配置
export CUDA_HOME=/usr/local/cuda
export NCCL_HOME=/path/to/nccl
export MPI_HOME=/usr/local

# 库路径
export LD_LIBRARY_PATH=$NCCL_HOME/lib:$MPI_HOME/lib:$LD_LIBRARY_PATH
```

## 5. 使用方法 / Usage

### 5.1 命令行参数 / Command Line Arguments

```bash
./simple_sendrecv_latency [options]

选项说明：
  -w <num>   预热迭代次数（默认：100）
  -n <num>   测量迭代次数（默认：1000）
  -b <size>  最小消息大小，字节（默认：8）
  -e <size>  最大消息大小，字节（默认：1MB）
  -h         显示帮助信息
```

### 5.2 典型使用场景 / Typical Use Cases

#### 场景1：基础延迟测试

```bash
# 使用默认参数进行测试
./simple_sendrecv_latency
```

#### 场景2：小消息延迟测试

```bash
# 测试8字节到1KB的小消息
./simple_sendrecv_latency -b 8 -e 1024 -n 10000
```

#### 场景3：大消息带宽测试

```bash
# 测试1MB到100MB的大消息
./simple_sendrecv_latency -b 1048576 -e 104857600 -n 100
```

#### 场景4：稳定性测试

```bash
# 长时间运行测试延迟稳定性
./simple_sendrecv_latency -w 1000 -n 100000
```

### 5.3 运行脚本说明 / Run Script Details

`run.sh` 提供了预配置的运行环境：

```bash
# NCCL性能调优参数
export NCCL_SHM_DISABLE=1    # 禁用共享内存，测试纯网络性能
export NCCL_P2P_DISABLE=1    # 禁用P2P，强制使用网络
export NCCL_IB_HCA=mlx5      # 指定InfiniBand HCA
export NCCL_DEBUG=INFO       # 调试信息级别

# MPI运行配置
mpirun -np 2 \
  -x CUDA_VISIBLE_DEVICES \
  -x LD_LIBRARY_PATH \
  -x NCCL_* \
  ./simple_sendrecv_latency
```
