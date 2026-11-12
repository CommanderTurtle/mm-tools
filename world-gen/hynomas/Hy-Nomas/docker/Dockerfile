# syntax=docker/dockerfile:1.7
#
# Build from the repository root:
#
#   docker build --network=host \
#     -f docker/Dockerfile -t hy-nomas:cuda13 .
#
# DeepGEMM, FlashAttention 2/3, and MagiAttention are compiled from pinned
# public releases so the build does not depend on private or prebuilt wheels.

ARG CUDA_IMAGE=nvidia/cuda:13.0.2-cudnn-devel-rockylinux8@sha256:f118c127be765aede48ad4d5b0289412acb37c68949a85dbfc211ddfd7fa94d1

FROM ${CUDA_IMAGE} AS python_runtime

ARG PYTHON_VERSION=3.13.11
ARG PYTHON_SHA256=03cfedbe06ce21bc44ce09245e091a77f2fee9ec9be5c52069048a181300b202
ARG TORCH_VERSION=2.10.0
ARG TORCHVISION_VERSION=0.25.0
ARG TORCHAUDIO_VERSION=2.10.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130

ENV CUDA_HOME=/usr/local/cuda \
    PATH=/opt/python3.13/bin:/opt/rh/gcc-toolset-13/root/usr/bin:/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/opt/python3.13/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# EPEL and PowerTools provide packages that are present in TencentOS by
# default but are not enabled in the upstream Rocky Linux 8 image.
RUN dnf install -y dnf-plugins-core epel-release \
    && dnf config-manager --set-enabled powertools \
    && dnf install -y \
        gcc-toolset-13 \
        bzip2-devel \
        curl \
        findutils \
        fuse-libs \
        gawk \
        gdbm-devel \
        git \
        libffi-devel \
        libibverbs \
        libibverbs-utils \
        libmnl \
        libusbx \
        libuuid-devel \
        libuv-devel \
        libxslt-devel \
        lsof \
        make \
        ncurses-devel \
        ninja-build \
        numactl-libs \
        openblas-devel \
        openssl-devel \
        pciutils \
        pdsh \
        pdsh-rcmd-ssh \
        rapidjson-devel \
        rdma-core \
        readline-devel \
        sqlite-devel \
        tar \
        tk-devel \
        unzip \
        wget \
        xz-devel \
        zlib-devel \
    && dnf clean all \
    && rm -rf /var/cache/dnf

RUN curl -fsSLO "https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz" \
    && echo "${PYTHON_SHA256}  Python-${PYTHON_VERSION}.tgz" | sha256sum -c - \
    && tar -xzf "Python-${PYTHON_VERSION}.tgz" \
    && cd "Python-${PYTHON_VERSION}" \
    && ./configure \
        --prefix=/opt/python3.13 \
        --enable-optimizations \
        --enable-shared \
        --with-ensurepip=install \
        --with-lto \
    && make -j"$(nproc)" \
    && make install \
    && cd / \
    && rm -rf "Python-${PYTHON_VERSION}" "Python-${PYTHON_VERSION}.tgz" \
    && ln -s /opt/python3.13/bin/python3.13 /usr/local/bin/python3 \
    && ln -s /opt/python3.13/bin/python3.13 /usr/local/bin/python \
    && ln -s /opt/python3.13/bin/pip3.13 /usr/local/bin/pip3 \
    && ln -s /opt/python3.13/bin/pip3.13 /usr/local/bin/pip \
    && echo /opt/python3.13/lib > /etc/ld.so.conf.d/python313.conf \
    && ldconfig

RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel packaging \
    && pip3 install --no-cache-dir \
        "torch==${TORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}" \
        "torchaudio==${TORCHAUDIO_VERSION}" \
        --index-url "${TORCH_INDEX_URL}" \
    && pip3 install --no-cache-dir \
        cmake==3.28.3 \
        ninja==1.13.2 \
        pybind11 \
        psutil

FROM python_runtime AS deepgemm_builder

ARG DEEPGEMM_VERSION=v2.1.1

# DeepGEMM ships the CUTLASS headers used by its runtime JIT inside the wheel.
# Force a local build so its extension matches this image's PyTorch/CUDA ABI.
RUN git clone --branch "${DEEPGEMM_VERSION}" --depth 1 \
        --recurse-submodules --shallow-submodules \
        https://github.com/deepseek-ai/DeepGEMM.git /tmp/DeepGEMM \
    && cd /tmp/DeepGEMM \
    && DG_FORCE_BUILD=1 DG_USE_LOCAL_VERSION=0 \
        python3 setup.py bdist_wheel -v

FROM python_runtime AS flash_attention_builder

ARG FLASH_ATTENTION_VERSION=v2.8.3.post2
ARG MAX_JOBS=8
ARG NVCC_THREADS=4

ENV FLASH_ATTENTION_FORCE_BUILD=TRUE \
    FLASH_ATTN_CUDA_ARCHS=90;100 \
    MAX_JOBS=${MAX_JOBS} \
    NVCC_THREADS=${NVCC_THREADS}

RUN git clone --branch "${FLASH_ATTENTION_VERSION}" --depth 1 \
        --recurse-submodules --shallow-submodules \
        https://github.com/Dao-AILab/flash-attention.git /tmp/flash-attention \
    && cd /tmp/flash-attention \
    && python3 setup.py bdist_wheel -v \
    && cd hopper \
    && python3 setup.py bdist_wheel -v

FROM python_runtime AS magi_attention_builder

ARG MAGI_ATTENTION_VERSION=v1.1.1
ARG MAX_JOBS=8
ARG NVCC_THREADS=4

ENV MAGI_ATTENTION_BUILD_COMPUTE_CAPABILITY=90,100 \
    MAGI_ATTENTION_PREBUILD_FFA=0 \
    MAX_JOBS=${MAX_JOBS} \
    NVCC_THREADS=${NVCC_THREADS}

RUN pip3 install --no-cache-dir \
        "nvidia-cutlass-dsl[cu13]==4.8.0" \
        nvidia-nvshmem-cu13==3.8.0 \
        versioningit==3.3.0 \
    && git clone --branch "${MAGI_ATTENTION_VERSION}" --depth 1 \
        --recurse-submodules --shallow-submodules \
        https://github.com/SandAI-org/MagiAttention.git /tmp/MagiAttention \
    && cd /tmp/MagiAttention \
    && python3 setup.py bdist_wheel -v

FROM python_runtime AS hy_nomas

ARG MAX_JOBS=8

ENV MAX_JOBS=${MAX_JOBS} \
    MAGI_ATTENTION_WORKSPACE_BASE=/workspace/.runtime/magi \
    LD_LIBRARY_PATH=/opt/python3.13/lib/python3.13/site-packages/torch/lib:${LD_LIBRARY_PATH}

COPY --from=deepgemm_builder /tmp/DeepGEMM/dist/*.whl /tmp/deepgemm/
COPY --from=flash_attention_builder /tmp/flash-attention/dist/*.whl /tmp/flash-attention/
COPY --from=flash_attention_builder /tmp/flash-attention/hopper/dist/*.whl /tmp/flash-attention/
COPY --from=magi_attention_builder /tmp/MagiAttention/dist/*.whl /tmp/magi-attention/

RUN pip3 install --no-cache-dir /tmp/deepgemm/*.whl \
    && pip3 install --no-cache-dir --no-deps \
        /tmp/flash-attention/*.whl \
        /tmp/magi-attention/*.whl \
    && rm -rf /tmp/deepgemm /tmp/flash-attention /tmp/magi-attention

# Optional dependencies from pyproject.toml, except torch (already installed
# from the CUDA 13.0 index) and opencv-python (headless is sufficient here).
RUN pip3 install --no-cache-dir \
        accelerate==1.10.1 \
        build \
        cuda-tile==1.3.0 \
        debugpy==1.8.22 \
        deepspeed==0.19.0 \
        diffusers==0.36.0 \
        einops==0.8.0 \
        flash-linear-attention \
        ftfy==6.3.0 \
        "hydra-core>=1.3" \
        liger-kernel \
        "nvidia-cutlass-dsl[cu13]==4.8.0" \
        nvidia-ml-py \
        nvidia-nvshmem-cu13==3.8.0 \
        numpy==2.1.3 \
        omegaconf==2.3.0 \
        opencv-python-headless==4.12.0.88 \
        pandas==2.3.3 \
        peft==0.18.0 \
        pillow==10.4.0 \
        pytest \
        qwen_vl_utils \
        setuptools_scm \
        sortedcontainers \
        timm==1.0.12 \
        tokenizers==0.22.1 \
        tomli-w \
        tqdm==4.67.1 \
        transformers==4.57.1 \
        tyro \
        versioningit==3.3.0 \
        wandb

WORKDIR /workspace/Hy-Nomas
COPY . .

RUN pip3 install --no-cache-dir . \
    && mkdir -p "${MAGI_ATTENTION_WORKSPACE_BASE}"

CMD ["/bin/bash"]
