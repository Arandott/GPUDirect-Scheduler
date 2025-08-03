# GPUDirect-Scheduler

## 项目概述

GPUDirect-Scheduler 是一个用于研究GPU间通信竞争（contention）的实验框架。本项目旨在通过系统化的基准测试和调度策略研究，深入理解和优化GPU集群中的通信性能。

### 核心目标 / Core Objectives

1. **通信性能分析** - 测量和分析GPU间通信的延迟特性
2. **带宽竞争研究** - 研究多GPU并发访问时的带宽竞争行为
3. **调度策略优化** - 开发智能调度算法以最小化通信竞争
4. **性能基准测试** - 提供标准化的测试工具集

### 项目架构 / Project Architecture

```
GPUDirect-Scheduler/
├── README.md                   # 项目主文档
├── docs/                       # 详细技术文档
│   ├── ARCHITECTURE.md         # 系统架构设计
│   ├── SENDRECV_UTIL.md       # Send/Recv工具详解
│   ├── TRIAD_UTIL.md          # Triad工具详解
│   └── DEVELOPMENT.md         # 开发指南
└── utils/                     # 核心工具集
    ├── sendrecv_util/         # NCCL通信延迟测试
    │   ├── simple_sendrecv_latency.cu
    │   ├── build.sh
    │   └── run.sh
    └── triad_util/            # GPU内存带宽测试
        ├── triad.cu
        ├── build.sh
        └── run.sh
```

## Quick Start

### Prerequisites

- **CUDA**: >= 12.2
- **NCCL**: >=2.25
- **MPI**: OpenMPI
- **GPU**: 支持GPUDirect的NVIDIA GPU（推荐 Ampere+）
- **操作系统**: Linux（Ubuntu 22.04+）
- 网卡：CX-5+·

### 编译和运行 / Build and Run

#### 1. NCCL Send/Recv 延迟测试

```bash
# 编译
cd utils/sendrecv_util
./build.sh

# 运行（使用MPI）
./run.sh
```

#### 2. GPU内存带宽测试（Triad）

```bash
# 编译
cd utils/triad_util
./build.sh

# 运行
./run.sh
```

## 工具简介

### sendrecv_util - NCCL通信延迟测试工具

该工具用于精确测量GPU间点对点通信的延迟特性，包括P50、P90、P95、P99等统计指标。

**主要特性**：

- 目前只支持两个rank之间互发消息
- 高精度延迟测量
- 支持不同消息大小的测试
- 详细的统计分析（百分位数、标准差等）

**使用示例**：

```bash
# 基础测试
./simple_sendrecv_latency

# 自定义参数
./simple_sendrecv_latency -w 1000 -n 10000 -b 64 -e 1048576
# -w: 预热迭代次数
# -n: 测量迭代次数  
# -b: 最小消息大小（字节）
# -e: 最大消息大小（字节）
```

### triad_util - GPU内存带宽测试工具

基于STREAM Triad基准测试，用于测量GPU内存带宽和评估内存访问竞争。

**主要特性**：

- 标准Triad操作：a[i] = b[i] + alpha * c[i]
- 实时带宽监控

**使用示例**：

```bash
# 默认运行（256M元素，100次迭代）
./triad

# 自定义参数
./triad 1073741824 1000  # 1G元素，1000次迭代
```
