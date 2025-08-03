#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# build.sh —— 编译 triad.cu，生成可执行文件 triad
# -----------------------------------------------------------------------------
set -euo pipefail               # 脚本出错立即退出，管道错误可被捕获

SRC=triad.cu
OUT=triad
ARCH=sm_90
OPT="-O3"
DEBUG="-lineinfo"               # 保留行号信息，便于后续 ncu 对应源码

echo "==>  nvcc $SRC  ->  $OUT  (arch=${ARCH})"
nvcc $SRC -o $OUT \
     $OPT $DEBUG \

echo "✅  完成编译，生成可执行文件 ./${OUT}"

