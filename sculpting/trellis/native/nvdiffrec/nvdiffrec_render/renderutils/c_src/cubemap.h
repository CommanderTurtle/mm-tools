
#pragma once

#include "common.h"

struct DiffuseCubemapKernelParams
{
    Tensor  cubemap;
    Tensor  out;
    dim3    gridSize;
};

struct SpecularCubemapKernelParams
{
    Tensor  cubemap;
    Tensor  bounds;
    Tensor  out;
    dim3    gridSize;
    float   costheta_cutoff;
    float   roughness;
};

struct SpecularBoundsKernelParams
{
    float   costheta_cutoff;
    Tensor  out;
    dim3    gridSize;
};
