/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 */

#include "./renderer.norm_cam_gradients.device.h"

namespace pulsar {
namespace Renderer {

template GLOBAL void norm_cam_gradients<ISONDEVICE>(Renderer renderer);

} // namespace Renderer
} // namespace pulsar
