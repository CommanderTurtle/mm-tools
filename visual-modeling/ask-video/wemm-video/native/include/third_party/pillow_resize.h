#pragma once
// Extracted without numerical changes from the pinned llama.cpp mtmd-image.cpp.
// Original implementation adapts Pillow Resample.c. See LICENSE.llama.cpp.
#include "clip-impl.h"
#include "clip-model.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>
struct wemm_pillow_resize {
    static bool resize(
            const clip_image_u8 & img,
            clip_image_u8 & dst,
            int target_width,
            int target_height,
            resize_algo algo) {
        // Fixed-point precision: 22 bits = 32 (int32_t) - 8 (uint8_t pixels) - 2 (headroom for accumulation)
        // This allows encoding fractional weights as integers: weight * 2^22
        const int PRECISION_BITS = 32 - 8 - 2;

        // Filter support radius
        double filter_support;
        switch (algo) {
            case RESIZE_ALGO_BILINEAR: filter_support = 1.0; break;
            case RESIZE_ALGO_BICUBIC:  filter_support = 2.0; break;
            case RESIZE_ALGO_LANCZOS:  filter_support = 3.0; break;
            default:
                throw std::runtime_error("Unsupported resize algorithm");
        }

        // Returns filter weight for distance x from pixel center
        // Note: for bicubic, Pillow uses a = -0.5 while GGML/PyTorch use a = -0.75
        auto resample_filter = [algo](double x) -> double {
            if (algo == RESIZE_ALGO_LANCZOS) {
                if (-3.0 <= x && x < 3.0) {
                    auto sinc = [](double v) {
                        if (v == 0.0) {
                            return 1.0;
                        }
                        const double pi_v = v * 3.141592653589793238462643383279502884;
                        return std::sin(pi_v) / pi_v;
                    };
                    return sinc(x) * sinc(x / 3.0);
                }
                return 0.0;
            }

            if (x < 0.0) {
                x = -x;
            }

            if (algo == RESIZE_ALGO_BILINEAR) {
                return x < 1.0 ? 1.0 - x : 0.0;
            }

            constexpr double a = -0.5;
            if (x < 1.0) {
                return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
            }
            if (x < 2.0) {
                return (((x - 5) * x + 8) * x - 4) * a;
            }
            return 0.0;  // Zero outside [-2, 2]
        };

        // Clipping function for 8-bit values
        auto clip8 = [](int val) -> uint8_t {
            if (val < 0) return 0;
            if (val > 255) return 255;
            return static_cast<uint8_t>(val);
        };

        // Precompute filter coefficients for ONE dimension (horizontal or vertical)
        //
        // Parameters:
        //   inSize  - Number of pixels in input dimension (e.g., src_width or src_height)
        //   outSize - Number of pixels in output dimension (e.g., target_width or target_height)
        //   bounds  - [OUTPUT] Array of size outSize*2 storing input pixel ranges:
        //             bounds[xx*2+0] = first input pixel index for output pixel xx (xmin)
        //             bounds[xx*2+1] = number of input pixels for output pixel xx (xcnt)
        //   weights - [OUTPUT] Array of size outSize*ksize storing fixed-point filter weights:
        //             kk[xx*ksize + x] = weight for input pixel x contributing to output pixel xx
        //
        // Returns: kernel size (ksize) - number of input pixels that contribute to each output pixel
        auto precompute_weights = [&](int inSize, int outSize,
                                     std::vector<int> & bounds, std::vector<int32_t> & weights) -> int {
            GGML_ASSERT(inSize > 0 && outSize > 0);
            double support, scale, filterscale;
            double center, ww, ss;
            int xx, x, ksize, xmin, xmax;

            // Calculate scaling factor: ratio of input range to output size
            filterscale = scale = static_cast<double>(inSize) / outSize;
            // For upsampling (scale < 1), keep filterscale = 1 to maintain filter sharpness
            // For downsampling (scale > 1), widen filter to prevent aliasing
            if (filterscale < 1.0) {
                filterscale = 1.0;
            }

            // Determine filter support radius and kernel size
            support = filter_support * filterscale;  // Widen filter when downsampling
            ksize = static_cast<int>(std::ceil(support)) * 2 + 1;  // Total pixels in kernel

            std::vector<double> pre_weights((size_t) outSize * ksize);  // Temporary weights
            bounds.resize((size_t) outSize * 2);


            // For each output pixel, compute its filter coefficients
            for (xx = 0; xx < outSize; xx++) {
                // Calculate the center position in input space (pixel-center convention: +0.5)
                center = (xx + 0.5) * scale;
                ww = 0.0;  // Sum of weights for normalization
                ss = 1.0 / filterscale;  // Scale factor for filter function

                // Determine the range of input pixels that contribute to this output pixel
                xmin = static_cast<int>(center - support + 0.5);
                if (xmin < 0) {
                    xmin = 0;
                }

                xmax = static_cast<int>(center + support + 0.5);
                if (xmax > inSize) {
                    xmax = inSize;
                }

                xmax -= xmin;

                // Compute filter weights for each contributing input pixel
                for (x = 0; x < xmax; x++) {
                    // Distance from input pixel center to output pixel center in input space
                    double w = resample_filter((x + xmin - center + 0.5) * ss);
                    pre_weights[(size_t) xx * ksize + x] = w;
                    ww += w;  // Accumulate for normalization
                }

                // Normalize weights to sum to 1.0 (preserves brightness)
                for (x = 0; x < xmax; x++) {
                    if (ww != 0.0) {
                        pre_weights[(size_t) xx * ksize + x] /= ww;
                    }
                }

                // Zero-pad remaining kernel positions
                for (; x < ksize; x++) {
                    pre_weights[(size_t) xx * ksize + x] = 0;
                }

                // Store input pixel range for this output pixel
                bounds[xx * 2 + 0] = xmin;
                bounds[xx * 2 + 1] = xmax;
            }

            // Convert floating-point coefficients to fixed-point integers
            // Formula: int32 = round(float * 2^PRECISION_BITS)
            weights.resize((size_t) outSize * ksize);

            const double fxp_scale = std::ldexp(1.0, PRECISION_BITS); // 1.0 * 2^PRECISION_BITS

            for (size_t i = 0; i < (size_t) outSize * ksize; i++) {
                // Pillow adds +/- 0.5 then truncates toward zero; std::round would round twice
                const double rounded = pre_weights[i] * fxp_scale + (pre_weights[i] < 0 ? -0.5 : 0.5);
                weights[i] = static_cast<int32_t>(rounded);
            }

            return ksize;
        };

        // Horizontal resampling pass
        // Resizes width from src to out_nx, preserving height
        auto resample_horizontal = [&](const uint8_t * src, int in_nx, int in_ny,
                                       int out_nx,
                                       int ksize, const std::vector<int> & bounds, const std::vector<int32_t> & weights) {
            std::vector<uint8_t> out((size_t) out_nx * in_ny * 3);

            // Process each row independently
            for (int yy = 0; yy < in_ny; yy++) {
                const uint8_t * src_row = src + (size_t) yy * in_nx * 3;
                uint8_t * dst_row = out.data() + (size_t) yy * out_nx * 3;

                // For each output pixel in this row
                for (int xx = 0; xx < out_nx; xx++) {
                    const int xmin = bounds[xx * 2 + 0];  // First input pixel index
                    const int xcnt = bounds[xx * 2 + 1];  // Number of input pixels
                    const int32_t * k = &weights[xx * ksize];
                    const uint8_t * p = src_row + (size_t) xmin * 3;

                    // Accumulators for RGB channels, with rounding bias (0.5 in fixed-point)
                    int32_t ss0 = 1 << (PRECISION_BITS - 1);
                    int32_t ss1 = 1 << (PRECISION_BITS - 1);
                    int32_t ss2 = 1 << (PRECISION_BITS - 1);

                    // Convolve: sum weighted input pixels
                    for (int x = 0; x < xcnt; x++) {
                        ss0 += p[0] * k[x];
                        ss1 += p[1] * k[x];
                        ss2 += p[2] * k[x];
                        p += 3;
                    }

                    // Convert back from fixed-point (divide by 2^PRECISION_BITS) and clamp to [0,255]
                    dst_row[xx * 3 + 0] = clip8(ss0 >> PRECISION_BITS);
                    dst_row[xx * 3 + 1] = clip8(ss1 >> PRECISION_BITS);
                    dst_row[xx * 3 + 2] = clip8(ss2 >> PRECISION_BITS);
                }
            }

            return out;
        };

        // Vertical resampling pass
        // Resizes height from src to out_ny, preserving width
        // Accumulates whole rows at once (contiguous access, auto-vectorizes well)
        auto resample_vertical = [&](const uint8_t * src, int in_nx,
                                     int out_ny,
                                     int ksize, const std::vector<int> & bounds, const std::vector<int32_t> & weight) {
            const size_t row_elems = (size_t) in_nx * 3;
            std::vector<uint8_t> out(row_elems * out_ny);
            std::vector<int32_t> acc(row_elems);

            // For each output row
            for (int yy = 0; yy < out_ny; yy++) {
                const int ymin = bounds[yy * 2 + 0];  // First input row index
                const int ycnt = bounds[yy * 2 + 1];  // Number of input rows
                const int32_t * k = &weight[yy * ksize];

                // Rounding bias (0.5 in fixed-point)
                std::fill(acc.begin(), acc.end(), 1 << (PRECISION_BITS - 1));

                // Convolve: accumulate each weighted input row
                for (int y = 0; y < ycnt; y++) {
                    const uint8_t * src_row = src + (size_t) (ymin + y) * row_elems;
                    const int32_t w = k[y];
                    for (size_t i = 0; i < row_elems; i++) {
                        acc[i] += src_row[i] * w;
                    }
                }

                // Convert back from fixed-point and clamp to [0,255]
                uint8_t * dst_row = out.data() + (size_t) yy * row_elems;
                for (size_t i = 0; i < row_elems; i++) {
                    dst_row[i] = clip8(acc[i] >> PRECISION_BITS);
                }
            }

            return out;
        };

        // Main resampling logic using separable two-pass approach
        const int src_width  = img.get_size().width;
        const int src_height = img.get_size().height;

        // sanity check on the target size
        if (target_width <= 0 || target_width > 65536 || target_height <= 0 || target_height > 65536) {
            throw std::runtime_error("resize target " + std::to_string(target_width) + "x" +
                                     std::to_string(target_height) + " is out of range (max 65536)");
        }

        bool need_horizontal = (target_width != src_width);
        bool need_vertical = (target_height != src_height);

        // Precompute filter coefficients for both dimensions
        std::vector<int> bounds_horiz, bounds_vert;
        std::vector<int32_t> weights_horiz, weights_vert;
        int ksize_horiz = 0, ksize_vert = 0;

        if (need_horizontal) {
            ksize_horiz = precompute_weights(src_width, target_width, bounds_horiz, weights_horiz);
        }

        if (need_vertical) {
            ksize_vert = precompute_weights(src_height, target_height, bounds_vert, weights_vert);
        }

        // Perform two-pass resampling
        const uint8_t * src = img.get_ro_buf().data();
        if (need_horizontal && need_vertical) {
            auto temp = resample_horizontal(src, src_width, src_height, target_width, ksize_horiz, bounds_horiz, weights_horiz);
            dst.set_size({target_width, target_height}, false);
            dst.cpy_buf(resample_vertical(temp.data(), target_width, target_height, ksize_vert, bounds_vert, weights_vert));
        } else if (need_horizontal) {
            dst.set_size({target_width, src_height}, false);
            dst.cpy_buf(resample_horizontal(src, src_width, src_height, target_width, ksize_horiz, bounds_horiz, weights_horiz));
        } else if (need_vertical) {
            dst.set_size({src_width, target_height}, false);
            dst.cpy_buf(resample_vertical(src, src_width, target_height, ksize_vert, bounds_vert, weights_vert));
        } else {
            // No resizing needed - direct copy
            dst.set_size(img.get_size(), false);
            dst.cpy_buf(img.get_ro_buf());
        }

        return true;
    }
};
