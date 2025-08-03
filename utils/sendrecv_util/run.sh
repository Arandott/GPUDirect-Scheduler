#!/usr/bin/env bash
set -euo pipefail

BIN=./simple_sendrecv_latency
[[ -x $BIN ]] || { echo "❌  $BIN not found – 先 ./build.sh"; exit 1; }

# ===== 在这里设 NCCL / MPI 路径 =================================================
NCCL_HOME=${NCCL_HOME:-/nlp_group/chenbaowen/documents_for_study/contention/nccl/build}
MPI_HOME=${MPI_HOME:-/usr/local}

export LD_LIBRARY_PATH="$NCCL_HOME/lib:$MPI_HOME/lib:${LD_LIBRARY_PATH:-}"
# ===============================================================================

# ───────── rendez-vous 变量 & NCCL 通道设置 ─────────
export MASTER_ADDR=${MASTER_ADDR:-10.82.104.19}
export MASTER_PORT=${MASTER_PORT:-8389}
export NCCL_SHM_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5}
export NCCL_DEBUG=${NCCL_DEBUG:-DEBUG}
export CUDA_VISIBLE_DEVICES=0,1

# 基准参数（字节数）
WARMUP=${WARMUP:-100}
ITERS=${ITERS:-10000}
MIN_BYTES=${MIN_BYTES:-8}
MAX_BYTES=${MAX_BYTES:-8192}

echo "=== NCCL latency test  (2 ranks via MPI, NET-only) ==="
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo

mpirun --allow-run-as-root -np 2 \
  -x CUDA_VISIBLE_DEVICES \
  -x LD_LIBRARY_PATH \
  -x NCCL_SHM_DISABLE -x NCCL_P2P_DISABLE -x NCCL_IB_HCA -x NCCL_DEBUG \
  -x MASTER_ADDR -x MASTER_PORT \
  "$BIN" -w "$WARMUP" -n "$ITERS" -b "$MIN_BYTES" -e "$MAX_BYTES"

