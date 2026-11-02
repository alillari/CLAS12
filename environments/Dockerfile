ARG BASE_IMAGE=nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04@sha256:7d2f6a8c2071d911524f95061a0db363e24d27aa51ec831fcccf9e76eb72bc92
FROM ${BASE_IMAGE}

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-dev build-essential git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/clas12
ENV PATH=/opt/clas12/bin:/usr/local/cuda/bin:$PATH \
    CUDA_HOME=/usr/local/cuda \
    TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
    PYTHONDONTWRITEBYTECODE=1
ARG TORCH_CUDA_ARCH_LIST
ARG MAX_JOBS=2
COPY environments /opt/clas12-recipe
RUN test -n "$TORCH_CUDA_ARCH_LIST" \
    && TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" MAX_JOBS="$MAX_JOBS" \
       bash /opt/clas12-recipe/install.sh
WORKDIR /workspace
CMD ["bash"]
