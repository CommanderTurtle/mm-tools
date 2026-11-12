# Ming-3.0-tts-0.5b inference image.
#
# torch 2.6 / CUDA 12.4 / flash-attn, trimmed to what infer.py imports, with
# transformers==4.46.3. For Ming-3.0-tts-tiny, install requirements-tiny.txt
# in a separate Python 3.10.15 environment with transformers==4.46.3.
#
#   docker build -f docker/Dockerfile -t ming-tts .
#
# Tested target: NVIDIA H800 / H20, CUDA 12.4.

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

RUN sed -i 's|http://archive.ubuntu.com/ubuntu|https://mirrors.aliyun.com/ubuntu|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com/ubuntu|https://mirrors.aliyun.com/ubuntu|g' /etc/apt/sources.list && \
    apt-get clean && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        wget \
        vim \
        libsndfile1 \
        build-essential \
        python3-dev \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/pip3.11 /usr/bin/pip

RUN pip install --no-cache-dir --upgrade pip && \
    pip config set global.index-url https://pypi.mirrors.ustc.edu.cn/simple/

# Base image already ships torch; re-pin so torchaudio matches.
RUN pip install --no-cache-dir \
    torch==2.6.0 \
    torchaudio==2.6.0 \
    torchvision==0.21.0

ENV FORCE_CUDA="1"
# A100: 8.0 · RTX 30xx: 8.6 · RTX 40xx / H100 / H800: 8.9 or 9.0
ENV TORCH_CUDA_ARCH_LIST="8.0;9.0"
ENV GROUPED_GEMM_CUTLASS="1"
ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_ROOT=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

RUN pip install --no-cache-dir --no-build-isolation \
    transformers==4.46.3 \
    tokenizers \
    accelerate==1.3.0 \
    huggingface_hub \
    numpy \
    soundfile==0.12.1 \
    tqdm \
    loguru \
    diffusers==0.33.0 \
    x_transformers \
    onnxruntime \
    inflect \
    einops \
    packaging \
    modelscope \
    fla-core==0.4.1 \
    atorch==1.7.0

# Text normalization (FST). Optional at runtime: --no_text_normalization.
RUN pip install --no-cache-dir pynini==2.1.6 || \
    echo "WARN: pynini wheel unavailable; use --no_text_normalization"

# flash-attn wheel for torch 2.6, CUDA 12, Python 3.11 and the pre-CXX11 ABI.
# Inference falls back to SDPA without flash-attn.
ARG FLASH_ATTN_WHEEL=flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
ARG FLASH_ATTN_URL=https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/${FLASH_ATTN_WHEEL}
RUN wget -q ${FLASH_ATTN_URL} -O /tmp/${FLASH_ATTN_WHEEL} && \
    pip install --no-cache-dir /tmp/${FLASH_ATTN_WHEEL} && \
    rm /tmp/${FLASH_ATTN_WHEEL}

WORKDIR /app

CMD ["/bin/bash"]
