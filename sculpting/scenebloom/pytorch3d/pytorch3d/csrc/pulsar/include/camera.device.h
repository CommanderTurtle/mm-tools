/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 */

#ifndef PULSAR_NATIVE_INCLUDE_CAMERA_DEVICE_H_
#define PULSAR_NATIVE_INCLUDE_CAMERA_DEVICE_H_

#include "../global.h"
#include "./camera.h"
#include "./commands.h"

namespace pulsar {
IHD CamGradInfo::CamGradInfo(int x) {
  cam_pos = make_float3(0.f, 0.f, 0.f);
  pixel_0_0_center = make_float3(0.f, 0.f, 0.f);
  pixel_dir_x = make_float3(0.f, 0.f, 0.f);
  pixel_dir_y = make_float3(0.f, 0.f, 0.f);
}
} // namespace pulsar

#endif
