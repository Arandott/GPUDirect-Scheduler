#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# run.sh —— 运行 (并可选 profile) triad
#
# 用法：
#   ./run.sh                       # 默认在 GPU0 上跑 triad
#   ./run.sh 1                     # 在 GPU1 上跑
#   ./run.sh 0 ncu                 # 在 GPU0 上跑并用 ncu 采样一次 triad kernel
# -----------------------------------------------------------------------------
set -euo pipefail

GPU_ID="${1:-0}"                 # 第一个位置参数：要使用的 GPU 逻辑序号，默认 0
MODE="${2:-run}"                 # 第二个参数：run（默认）或 ncu（profile）
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
echo "🔧  已锁定 CUDA_VISIBLE_DEVICES=${GPU_ID}"

######################### 需要填的 UUID #########################
TRIAD_CI_UUID="MPS-UUID-OF-50P-CI"   # <<<  ← 用实际 UUID 替换
################################################################

export MPS_VISIBLE_DEVICES="$TRIAD_CI_UUID" # 只看得到 70% CI

BIN=./triad                      # 可执行文件路径
# ----------------------------- triad 运行参数 -----------------------------
ELEMS=$((1024*1024*1024))        # 这里假设 triad.cu 默认为 float *4 字节，1G elements≈4 GiB
BLOCKS=131072                    # 你之前代码里的 gridSize（block 数）
THREADS=1024                     # 代码里的 blockDim
ITERS=100                        # 内部循环次数 (loops)
# 如 triad.cu 改为解析命令行，可把参数改成 "$1 $2 ..." 方式传入
ARGS="${ELEMS} ${BLOCKS} ${THREADS} ${ITERS}"
# ----------------------------------------------------------------------

if [[ "${MODE}" == "ncu" ]]; then
  echo "🕵️  使用 Nsight Compute 采样 1 次 triad kernel"
  # -f             覆盖同名 .ncu-rep
  # -o triad_bw    输出 triad_bw.ncu-rep
  # --set full     捕获所有 section；后续可在 GUI 里筛选
  # -c 1           只捕获一次 kernel（triad 在迭代循环里可能 launch 多次）
  ncu -f -o triad_bw --set full -c 1 "${BIN}" ${ARGS}
else
  echo "🚀  直接运行 triad"
  "${BIN}" ${ARGS}
fi

