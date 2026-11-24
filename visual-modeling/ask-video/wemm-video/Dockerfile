# syntax=docker/dockerfile:1
ARG CUDA_BASE_IMAGE=docker.io/nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04@sha256:1cb02388491c05441f8963d42bc38b6ec5462fba95a8e3bb2cb8e6091f0738c8
ARG WEMM_PROFILE=vllm-0.18.0
FROM ${CUDA_BASE_IMAGE} AS base

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get -o Acquire::Retries=3 update \
    && apt-get install -y --no-install-recommends \
        build-essential ca-certificates ffmpeg git libgl1 libglib2.0-0t64 \
        libnuma1 libsndfile1 ninja-build python3 python3-dev python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/wemm-env

ENV VIRTUAL_ENV=/opt/wemm-env \
    PATH=/opt/wemm-env/bin:${PATH} \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

FROM scratch AS hf-wheel-cache
FROM base AS hf-kernels
ARG WEMM_PROFILE
ARG HF_BUILD_JOBS=1
RUN --mount=type=bind,source=patch,target=/opt/wemm-patch \
    --mount=type=bind,from=hf-wheel-cache,target=/opt/hf-wheel-cache \
    --mount=type=cache,target=/var/cache/wemm-hf-wheels,id=wemm-fa2-2.8.3.post2-conv-1.7.0-torch-2.13-cu129,sharing=locked \
    --mount=type=cache,target=/root/.cache/uv,id=wemm-video-uv \
    --mount=type=cache,target=/root/.cache/pip,id=wemm-video-pip \
    HF_BUILD_JOBS="$HF_BUILD_JOBS" bash /opt/wemm-patch/build_hf_wheels.sh "$WEMM_PROFILE"

FROM base AS runtime
ARG WEMM_PROFILE
RUN --mount=type=bind,source=patch,target=/opt/wemm-patch \
    --mount=type=bind,from=hf-kernels,source=/opt/hf-wheels,target=/opt/hf-wheels \
    --mount=type=cache,target=/root/.cache/uv,id=wemm-video-uv \
    --mount=type=cache,target=/root/.cache/pip,id=wemm-video-pip \
    bash /opt/wemm-patch/install_locked.sh "$WEMM_PROFILE" \
    && python -c 'import torch; assert torch.version.cuda == "12.9"' \
    && python -c 'from torchcodec.decoders import AudioDecoder, VideoDecoder'

# Match the NVIDIA driver capabilities used by the reference images.
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

WORKDIR /workspace
ENTRYPOINT []
CMD ["/bin/bash"]
