#include "torch_common.inl"

//------------------------------------------------------------------------
// Python CudaRaster state wrapper.

namespace CR { class CudaRaster; }
class RasterizeCRStateWrapper
{
public:
    RasterizeCRStateWrapper     (int cudaDeviceIdx);
    ~RasterizeCRStateWrapper    (void);

    CR::CudaRaster*             cr;
    int                         cudaDeviceIdx;
};

//------------------------------------------------------------------------
// Mipmap wrapper to prevent intrusion from Python side.

class TextureMipWrapper
{
public:
    torch::Tensor               mip;
    int                         max_mip_level;
    std::vector<int64_t>        texture_size;   // For error checking.
    bool                        cube_mode;      // For error checking.
};


//------------------------------------------------------------------------
// Antialias topology hash wrapper to prevent intrusion from Python side.

class TopologyHashWrapper
{
public:
    torch::Tensor               ev_hash;
};

//------------------------------------------------------------------------
