#include <torch/extension.h>

#include "iou_box3d/iou_box3d.h"
#include "sample_farthest_points/sample_farthest_points.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("iou_box3d", &IoUBox3D);
  module.def("sample_farthest_points", &FarthestPointSampling);
}
