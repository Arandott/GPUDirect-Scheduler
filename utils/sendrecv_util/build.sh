#!/usr/bin/env bash
# ============================================================================
# build.sh  ——  Compile simple_sendrecv_latency.cu
# ============================================================================
set -euo pipefail

echo "===== Build: NCCL Send/Recv Latency Test ====="

# — 1. 环境检查 --------------------------------------------------------------
command -v nvcc  >/dev/null || { echo "nvcc not found, install CUDA"; exit 1; }
export MPI_HOME=/usr/local

CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
NCCL_HOME=${NCCL_HOME:-/nlp_group/chenbaowen/documents_for_study/contention/nccl/build}

[[ -f "$NCCL_HOME/include/nccl.h" ]] || {
  echo "NCCL headers missing in $NCCL_HOME/include – set NCCL_HOME"; exit 1; }

# — 2. 编译选项 --------------------------------------------------------------
NVCC="$CUDA_HOME/bin/nvcc"
CFLAGS="-O3 -std=c++11"
INCLUDES="-I$CUDA_HOME/include -I$NCCL_HOME/include -I$MPI_HOME/include"
LIBS="-L$CUDA_HOME/lib64 -L$NCCL_HOME/lib -lnccl -lcudart -L$MPI_HOME/lib -lmpi"

# MPI（可选）
if command -v mpicc &>/dev/null; then
  echo "MPI detected – enabling MPI support"
  CFLAGS+=" -DMPI_SUPPORT"
else
  echo "MPI not detected – build without MPI"
fi

# — 3. 编译 -----------------------------------------------------------------
echo "Compiling simple_sendrecv_latency.cu ..."
set -x
$NVCC $CFLAGS $INCLUDES simple_sendrecv_latency.cu -o simple_sendrecv_latency $LIBS
set +x
echo "✅ Build finished: ./simple_sendrecv_latency"

