NCCL_HOME=${NCCL_HOME:-/nlp_group/chenbaowen/documents_for_study/contention/nccl/build}
NVCC="$CUDA_HOME/bin/nvcc -ccbin $(which mpicxx)"
nvcc -O3 -std=c++11 \
     -I$NCCL_HOME/include -L$NCCL_HOME/lib -lnccl \
     simple_sendrecv_latency.cu -o simple_sendrecv_latency

