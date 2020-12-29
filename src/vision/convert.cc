#include "vision/convert.h"

namespace dlfm::vision {

template <typename T>
void convert_block_impl(
  T *x,
  T *y,
  int64_t height,
  int64_t width,
  int64_t x_channel,
  int64_t y_channel,
  const std::vector<size_t> &idxs,
  int64_t start,
  int64_t end) {
  for (int64_t h = start; h < end; ++h) {
    for (int64_t w = 0; w < width; ++w) {
      T *x_offset = x + h * width * x_channel + w * x_channel;
      T *y_offset = y + h * width * y_channel + w * y_channel;

      for (int64_t c = 0; c < y_channel; ++c) {
        y_offset[c] = x_offset[idxs[c]];
      }
    }
  }
}

template <typename T>
void convert_impl(
  Eigen::ThreadPoolDevice *eigen_device,
  T *x,
  T *y,
  int64_t height,
  int64_t width,
  int64_t x_channel,
  int64_t y_channel,
  const std::vector<size_t> &idxs) {
  int64_t num_threads = (int64_t)eigen_device->numThreads();
  int64_t block_size = (height + num_threads - 1) / num_threads;
  int64_t need_num_threads = (height + block_size - 1) / block_size;

  Eigen::Barrier barrier((unsigned int)(need_num_threads));

  for (int64_t i = 0; i < need_num_threads; ++i) {
    int64_t start_index = i * block_size;
    int64_t end_index = std::min<int64_t>(start_index + block_size, height);

    eigen_device->enqueue_with_barrier(
      &barrier,
      &convert_block_impl<T>,
      x,
      y,
      height,
      width,
      x_channel,
      y_channel,
      idxs,
      start_index,
      end_index);
  }

  barrier.Wait();
}

Tensor convert(const Tensor &x, std::vector<size_t> idx) {
  ARGUMENT_CHECK(3 == x.shape().ndims(), "convert need input dimension is 3.");

  int64_t height = x.shape()[0];
  int64_t width  = x.shape()[1];

  int64_t x_channel = x.shape()[2];
  int64_t y_channel = idx.size();

  for (auto i : idx) {
    ARGUMENT_CHECK(i >= 0 && i < x_channel, "idx out of range");
  }

  auto y = Tensor::create({height, width, y_channel}, x.element_type());

  if (x.element_type().is<float>()) {
    convert_impl(
      x.eigen_device().get(),
      x.data<float>(),
      y.data<float>(),
      height,
      width,
      x_channel,
      y_channel,
      idx);
  } else if (x.element_type().is<uint8_t>()) {
    convert_impl(
      x.eigen_device().get(),
      x.data<uint8_t>(),
      y.data<uint8_t>(),
      height,
      width,
      x_channel,
      y_channel,
      idx);
  } else {
    RUNTIME_ERROR("element type:" << x.element_type().name() << " not support!");
  }

  return y;
}

Tensor bgra_2_rgb(const Tensor &x) {
  ARGUMENT_CHECK(3 == x.shape().ndims(), "x ndims must be 3");
  ARGUMENT_CHECK(4 == x.shape()[2], "last dimension must be 4");

  return convert(x, {2, 1, 0});
}

Tensor bgra_2_rgba(const Tensor &x) {
  ARGUMENT_CHECK(3 == x.shape().ndims(), "x ndims must be 3");
  ARGUMENT_CHECK(4 == x.shape()[2], "last dimension must be 4");

  return convert(x, { 2, 1, 0, 3 });
}

Tensor rgbx_2_rgb(const Tensor &x) {
  ARGUMENT_CHECK(3 == x.shape().ndims(), "x ndims must be 3");
  ARGUMENT_CHECK(4 == x.shape()[2], "last dimension must be 4");

  return convert(x, { 0, 1, 2 });
}

//------------------------------------------------------------------------------------------------
// yuv 2 rgb.
void yuv_2_rgb_full_uint8_impl(Eigen::ThreadPoolDevice *eigen_device, uint8_t *y, uint8_t *uv, uint8_t *rgb, int64_t height, int64_t width) {
  auto block = [](uint8_t *y, uint8_t *uv, uint8_t *rgb, int64_t height, int64_t width, int64_t start, int64_t end) {
    for (int64_t h = start; h < end; ++h) {
      uint8_t *rgb_ptr = rgb + h * width * 3;
      uint8_t *y_ptr   = y   + h * width;
      uint8_t *uv_ptr  = uv  + (h / 2) * (width / 2) * 2;

      for (int64_t w = 0; w < width; ++w) {
        uint8_t yv = y_ptr[0];
        uint8_t uv = uv_ptr[0];
        uint8_t vv = uv_ptr[1];

        float r = yv                        + 1.4f  * (vv - 128.f);
        float g = yv - 0.213 * (uv - 128.f) - 0.711 * (vv - 128.f);
        float b = yv + 1.765 * (uv - 128.f);

        rgb_ptr[0] = uint8_t(std::clamp<float>(r, 0, 255));
        rgb_ptr[1] = uint8_t(std::clamp<float>(g, 0, 255));
        rgb_ptr[2] = uint8_t(std::clamp<float>(b, 0, 255));

        rgb_ptr += 3;
        y_ptr   += 1;
        uv_ptr  += 2 * (w % 2);
      }
    }
  };

  int64_t num_threads = (int64_t)eigen_device->numThreads();
  int64_t block_size = (height + num_threads - 1) / num_threads;
  int64_t need_num_threads = (height + block_size - 1) / block_size;

  Eigen::Barrier barrier((unsigned int)(need_num_threads));

  for (int64_t i = 0; i < need_num_threads; ++i) {
    int64_t start_index = i * block_size;
    int64_t end_index = std::min<int64_t>(start_index + block_size, height);

    eigen_device->enqueue_with_barrier(
      &barrier,
      block,
      y,
      uv,
      rgb,
      height,
      width,
      start_index,
      end_index);
  }

  barrier.Wait();
}

Tensor yuv_2_rgb_full(const Tensor &y, const Tensor &uv) {
  ARGUMENT_CHECK(3 == y.shape().ndims() && 1 == y.shape()[2], "y shape error");
  ARGUMENT_CHECK(3 == uv.shape().ndims() && 2 == uv.shape()[2], "uv shape error");
  ARGUMENT_CHECK(y.element_type() == uv.element_type(), "y uv element type must same");

  int64_t height = y.shape()[0];
  int64_t width  = y.shape()[1];

  ARGUMENT_CHECK(height == 2 * uv.shape()[0] && width == 2 * uv.shape()[1], "y/uv shape error");

  auto rgb = Tensor::create({height, width, 3}, y.element_type());

  if (y.element_type().is<uint8_t>()) {
    yuv_2_rgb_full_uint8_impl(y.eigen_device().get(), y.data<uint8_t>(), uv.data<uint8_t>(), rgb.data<uint8_t>(), height, width);
  } else {
    RUNTIME_ERROR("element type:" << y.element_type().name() << " not support!");
  }

  return rgb;
}

void yuv_2_rgb_video_uint8_impl(Eigen::ThreadPoolDevice *eigen_device, uint8_t *y, uint8_t *uv, uint8_t *rgb, int64_t height, int64_t width) {
  auto block = [](uint8_t *y, uint8_t *uv, uint8_t *rgb, int64_t height, int64_t width, int64_t start, int64_t end) {
    for (int64_t h = start; h < end; ++h) {
      uint8_t *rgb_ptr = rgb + h * width * 3;
      uint8_t *y_ptr   = y + h * width;
      uint8_t *uv_ptr  = uv + (h / 2) * (width / 2) * 2;

      for (int64_t w = 0; w < width; ++w) {
        uint8_t yv = y_ptr[0];
        uint8_t uv = uv_ptr[0];
        uint8_t vv = uv_ptr[1];

        float r = 1.164 * (yv - 16.f)                         + 1.596f * (vv - 128.f);
        float g = 1.164 * (yv - 16.f) - 0.392  * (uv - 128.f) - 0.813 * (vv - 128.f);
        float b = 1.164 * (yv - 16.f) + 2.017  * (uv - 128.f);

        rgb_ptr[0] = uint8_t(std::clamp<float>(r, 0, 255));
        rgb_ptr[1] = uint8_t(std::clamp<float>(g, 0, 255));
        rgb_ptr[2] = uint8_t(std::clamp<float>(b, 0, 255));

        rgb_ptr += 3;
        y_ptr   += 1;
        uv_ptr  += 2 * (w % 2);
      }
    }
  };

  int64_t num_threads = (int64_t)eigen_device->numThreads();
  int64_t block_size = (height + num_threads - 1) / num_threads;
  int64_t need_num_threads = (height + block_size - 1) / block_size;

  Eigen::Barrier barrier((unsigned int)(need_num_threads));

  for (int64_t i = 0; i < need_num_threads; ++i) {
    int64_t start_index = i * block_size;
    int64_t end_index = std::min<int64_t>(start_index + block_size, height);

    eigen_device->enqueue_with_barrier(
      &barrier,
      block,
      y,
      uv,
      rgb,
      height,
      width,
      start_index,
      end_index);
  }

  barrier.Wait();
}

Tensor yuv_2_rgb_video(const Tensor &y, const Tensor &uv) {
  ARGUMENT_CHECK(3 == y.shape().ndims() && 1 == y.shape()[2], "y shape error");
  ARGUMENT_CHECK(3 == uv.shape().ndims() && 2 == uv.shape()[2], "uv shape error");
  ARGUMENT_CHECK(y.element_type() == uv.element_type(), "y uv element type must same");

  int64_t height = y.shape()[0];
  int64_t width  = y.shape()[1];

  ARGUMENT_CHECK(height == 2 * uv.shape()[0] && width == 2 * uv.shape()[1], "y/uv shape error");

  auto rgb = Tensor::create({height, width, 3}, y.element_type());

  if (y.element_type().is<uint8_t>()) {
    yuv_2_rgb_video_uint8_impl(y.eigen_device().get(), y.data<uint8_t>(), uv.data<uint8_t>(), rgb.data<uint8_t>(), height, width);
  } else {
    RUNTIME_ERROR("element type:" << y.element_type().name() << " not support!");
  }

  return rgb;
}

void nv12_2_rgb_video_uint8_impl(
  Eigen::ThreadPoolDevice *eigen_device,
  uint8_t *y,
  uint8_t *uv,
  uint8_t *rgb,
  int64_t height,
  int64_t width) {
  // start_h and end_h must be divided by 2.
  auto block = [](uint8_t *y, uint8_t *uv, uint8_t *rgb, int64_t height, int64_t width, int64_t start_h, int64_t end_h) {
#if defined(__ARM_NEON__)
    const uint8x8_t y_shift  = vdup_n_u8(16);
    const int16x8_t half     = vdupq_n_u16(128);
    const int32x4_t rounding = vdupq_n_s32(128);

    uint8x8x3_t pblock;
#endif

    int64_t limit = width / 8 * 8;

    for (int64_t h = start_h; h < end_h; h += 2) {
      uint8_t *rgbp = rgb + h * width * 3;
      uint8_t *yp   = y   + h * width;
      uint8_t *uvp  = uv  + (h / 2) * width;

      int64_t w = 0;

#if defined(__ARM_NEON__)
      for (; w < limit; w += 8) {
        uint16x8_t yt = vmovl_u8(vqsub_u8(vld1_u8(yp + w), y_shift));
        const int32x4_t y00 = vmulq_n_u32(vmovl_u16(vget_low_u16(yt)), 298);
        const int32x4_t y01 = vmulq_n_u32(vmovl_u16(vget_high_u16(yt)), 298);

        yt = vmovl_u8(vqsub_u8(vld1_u8(yp + w + width), y_shift));
        const int32x4_t y10 = vmulq_n_u32(vmovl_u16(vget_low_u16(yt)), 298);
        const int32x4_t y11 = vmulq_n_u32(vmovl_u16(vget_high_u16(yt)), 298);

        // (u0, v0, u1, v1, u2, v2, u3, v3)
        int16x8_t uvt = vsubq_s16((int16x8_t)vmovl_u8(vld1_u8(uvp + w)), half);

        // uv_tiple.val[0] : u0, u1, u2, u3
        // uv_tiple.val[1] : v0, v1, v2, v3
        const int16x4x2_t uv_tuple = vuzp_s16(vget_low_s16(uvt), vget_high_s16(uvt));

        // rt : 128        + 409V
        // gt : 128 - 100U - 208V
        // bt : 128 + 516U
        const int32x4_t rt = vmlal_n_s16(rounding, uv_tuple.val[1], 409);
        const int32x4_t gt = vmlal_n_s16(vmlal_n_s16(rounding, uv_tuple.val[1], -208), uv_tuple.val[0], -100);
        const int32x4_t bt = vmlal_n_s16(rounding, uv_tuple.val[0], 516);

        const int32x4x2_t r = vzipq_s32(rt, rt); // [rt0, rt0, rt1, rt1] [ rt2, rt2, rt3, rt3]
        const int32x4x2_t g = vzipq_s32(gt, gt); // [gt0, gt0, gt1, gt1] [ gt2, gt2, gt3, gt3]
        const int32x4x2_t b = vzipq_s32(bt, bt); // [bt0, bt0, bt1, bt1] [ bt2, bt2, bt3, bt3]

        // upper 8 pixels
        pblock.val[0] = vshrn_n_u16(vcombine_u16(vqmovun_s32(vaddq_s32(r.val[0], y00)), vqmovun_s32(vaddq_s32(r.val[1], y01))), 8);
        pblock.val[1] = vshrn_n_u16(vcombine_u16(vqmovun_s32(vaddq_s32(g.val[0], y00)), vqmovun_s32(vaddq_s32(g.val[1], y01))), 8);
        pblock.val[2] = vshrn_n_u16(vcombine_u16(vqmovun_s32(vaddq_s32(b.val[0], y00)), vqmovun_s32(vaddq_s32(b.val[1], y01))), 8);

        vst3_u8(rgbp + w * 3, pblock);

        // lower 8 pixels
        pblock.val[0] = vshrn_n_u16(vcombine_u16(vqmovun_s32(vaddq_s32(r.val[0], y10)), vqmovun_s32(vaddq_s32(r.val[1], y11))), 8);
        pblock.val[1] = vshrn_n_u16(vcombine_u16(vqmovun_s32(vaddq_s32(g.val[0], y10)), vqmovun_s32(vaddq_s32(g.val[1], y11))), 8);
        pblock.val[2] = vshrn_n_u16(vcombine_u16(vqmovun_s32(vaddq_s32(b.val[0], y10)), vqmovun_s32(vaddq_s32(b.val[1], y11))), 8);

        vst3_u8(rgbp + (width + w) * 3, pblock);
      }
#endif

      for (; w < width; ++w) {
        uint8_t upper_y_val = yp[w];
        uint8_t lower_y_val = yp[w + width];

        uint8_t u_val = uvp[w / 2];
        uint8_t v_val = uvp[w / 2 + 1];

        int32_t r = (298 * (upper_y_val - 16)                       + 409 * (v_val - 128) + 128) >> 8;
        int32_t g = (298 * (upper_y_val - 16) - 100 * (u_val - 128) - 208 * (v_val - 128) + 128) >> 8;
        int32_t b = (298 * (upper_y_val - 16) + 516 * (u_val - 128)                       + 128) >> 8;

        rgbp[w * 3    ] = (uint8_t)(std::clamp<int32_t>(r, 0, 255));
        rgbp[w * 3 + 1] = (uint8_t)(std::clamp<int32_t>(g, 0, 255));
        rgbp[w * 3 + 2] = (uint8_t)(std::clamp<int32_t>(b, 0, 255));

        r = (298 * (lower_y_val - 16) + 409 * (v_val - 128) + 128) >> 8;
        g = (298 * (lower_y_val - 16) - 100 * (u_val - 128) - 208 * (v_val - 128) + 128) >> 8;
        b = (298 * (lower_y_val - 16) + 516 * (u_val - 128) + 128) >> 8;

        rgbp[w * 3 + width * 3    ] = (uint8_t)(std::clamp<int32_t>(r, 0, 255));
        rgbp[w * 3 + width * 3 + 1] = (uint8_t)(std::clamp<int32_t>(g, 0, 255));
        rgbp[w * 3 + width * 3 + 2] = (uint8_t)(std::clamp<int32_t>(b, 0, 255));
      }
    }
  };

  int64_t height_2 = height / 2;

  int64_t num_threads = (int64_t)eigen_device->numThreads();
  int64_t block_size = (height_2 + num_threads - 1) / num_threads;
  int64_t need_num_threads = (height_2 + block_size - 1) / block_size;

  Eigen::Barrier barrier((unsigned int)(need_num_threads));

  for (int64_t i = 0; i < need_num_threads; ++i) {
    int64_t start_h = i * block_size * 2;
    int64_t end_h = std::min<int64_t>(start_h + block_size * 2, height);

    eigen_device->enqueue_with_barrier(
      &barrier,
      block,
      y,
      uv,
      rgb,
      height,
      width,
      start_h,
      end_h);
  }

  barrier.Wait();
}

//To convert Y'UV to RGB :
//matrix from :
//|R|   | 298    0     409 | | Y'- 16  |
//|G| = | 298 - 100 -  208 | | U - 128 |
//|B|   | 298   516     0  | | V - 128 |
//then shift 8 bits, i.e.
//
//in integer math :
//R = clamp((298 * (Y'-16)              + 409*(V-128) + 128) >> 8)
//G = clamp((298 * (Y'-16)- 100*(U-128) - 208*(V-128) + 128) >> 8)
//B = clamp((298 * (Y'-16)+ 516*(U-128)               + 128) >> 8)
Tensor nv12_2_rgb_video(const Tensor &y, const Tensor &uv) {
  ARGUMENT_CHECK(3 == y.shape().ndims() && 1 == y.shape()[2], "y shape error");
  ARGUMENT_CHECK(3 == uv.shape().ndims() && 2 == uv.shape()[2], "uv shape error");
  ARGUMENT_CHECK(y.element_type() == uv.element_type(), "y uv element type must same");

  int64_t height = y.shape()[0];
  int64_t width = y.shape()[1];

  ARGUMENT_CHECK(height == 2 * uv.shape()[0] && width == 2 * uv.shape()[1], "y/uv shape error");

  auto rgb = Tensor::create({ height, width, 3 }, y.element_type());

  if (y.element_type().is<uint8_t>()) {
    nv12_2_rgb_video_uint8_impl(y.eigen_device().get(), y.data<uint8_t>(), uv.data<uint8_t>(), rgb.data<uint8_t>(), height, width);
  } else {
    RUNTIME_ERROR("element type:" << y.element_type().name() << " not support!");
  }

  return rgb;
}

void nv12_2_rgb_full_uint8_impl(
  Eigen::ThreadPoolDevice *eigen_device,
  uint8_t *y,
  uint8_t *uv,
  uint8_t *rgb,
  int64_t height,
  int64_t width) {
  // start_h and end_h must be divided by 2.
  auto block = [](uint8_t *y, uint8_t *uv, uint8_t *rgb, int64_t height, int64_t width, int64_t start_h, int64_t end_h) {
#if defined(__ARM_NEON__)
    const uint8x8_t y_shift = vdup_n_u8(16);
    const int16x8_t half = vdupq_n_u16(128);
    const int32x4_t rounding = vdupq_n_s32(128);

    uint8x8x3_t pblock;
#endif

    int64_t limit = width / 8 * 8;

    for (int64_t h = start_h; h < end_h; h += 2) {
      uint8_t *rgbp = rgb + h * width * 3;
      uint8_t *yp = y + h * width;
      uint8_t *uvp = uv + (h / 2) * width;

      int64_t w = 0;

#if defined(__ARM_NEON__)
      for (; w < limit; w += 8) {
        uint16x8_t yt = vmovl_u8(vld1_u8(yp + w));
        const int32x4_t y00 = vmovl_u16(vget_low_u16(yt));
        const int32x4_t y01 = vmovl_u16(vget_high_u16(yt));

        yt = vmovl_u8(vld1_u8(yp + w + width));
        const int32x4_t y10 = vmovl_u16(vget_low_u16(yt));
        const int32x4_t y11 = vmovl_u16(vget_high_u16(yt));

        // (u0, v0, u1, v1, u2, v2, u3, v3)
        int16x8_t uvt = vsubq_s16((int16x8_t)vmovl_u8(vld1_u8(uvp + w)), half);

        // uv_tiple.val[0] : u0, u1, u2, u3
        // uv_tiple.val[1] : v0, v1, v2, v3
        const int16x4x2_t uv_tuple = vuzp_s16(vget_low_s16(uvt), vget_high_s16(uvt));

        // rt :128 +        359V
        // gt :128 -   88U + 183V
        // bt :128 +  454U
        const int32x4_t rt = vmlal_n_s16(rounding, uv_tuple.val[1], 359);
        const int32x4_t gt = vmlal_n_s16(vmlal_n_s16(rounding, uv_tuple.val[1], 183), uv_tuple.val[0], -88);
        const int32x4_t bt = vmlal_n_s16(rounding, uv_tuple.val[0], 454);

        const int32x4x2_t r = vzipq_s32(rt, rt); // [rt0, rt0, rt1, rt1] [ rt2, rt2, rt3, rt3]
        const int32x4x2_t g = vzipq_s32(gt, gt); // [gt0, gt0, gt1, gt1] [ gt2, gt2, gt3, gt3]
        const int32x4x2_t b = vzipq_s32(bt, bt); // [bt0, bt0, bt1, bt1] [ bt2, bt2, bt3, bt3]

        // upper 8 pixels
        pblock.val[0] = vshrn_n_u16(vcombine_u16(vqmovun_s32(vaddq_s32(r.val[0], y00)), vqmovun_s32(vaddq_s32(r.val[1], y01))), 8);
        pblock.val[1] = vshrn_n_u16(vcombine_u16(vqmovun_s32(vaddq_s32(g.val[0], y00)), vqmovun_s32(vaddq_s32(g.val[1], y01))), 8);
        pblock.val[2] = vshrn_n_u16(vcombine_u16(vqmovun_s32(vaddq_s32(b.val[0], y00)), vqmovun_s32(vaddq_s32(b.val[1], y01))), 8);

        vst3_u8(rgbp + w * 3, pblock);

        // lower 8 pixels
        pblock.val[0] = vshrn_n_u16(vcombine_u16(vqmovun_s32(vaddq_s32(r.val[0], y10)), vqmovun_s32(vaddq_s32(r.val[1], y11))), 8);
        pblock.val[1] = vshrn_n_u16(vcombine_u16(vqmovun_s32(vaddq_s32(g.val[0], y10)), vqmovun_s32(vaddq_s32(g.val[1], y11))), 8);
        pblock.val[2] = vshrn_n_u16(vcombine_u16(vqmovun_s32(vaddq_s32(b.val[0], y10)), vqmovun_s32(vaddq_s32(b.val[1], y11))), 8);

        vst3_u8(rgbp + (width + w) * 3, pblock);
      }
#endif

      for (; w < width; ++w) {
        uint8_t upper_y_val = yp[w];
        uint8_t lower_y_val = yp[w + width];

        uint8_t u_val = uvp[w / 2];
        uint8_t v_val = uvp[w / 2 + 1];

        int32_t r = (upper_y_val                       + 359 * (v_val - 128) + 128) >> 8;
        int32_t g = (upper_y_val - 88  * (u_val - 128) + 183 * (v_val - 128) + 128) >> 8;
        int32_t b = (upper_y_val + 454 * (u_val - 128) +                       128) >> 8;

        rgbp[w * 3    ] = (uint8_t)(std::clamp<int32_t>(r, 0, 255));
        rgbp[w * 3 + 1] = (uint8_t)(std::clamp<int32_t>(g, 0, 255));
        rgbp[w * 3 + 2] = (uint8_t)(std::clamp<int32_t>(b, 0, 255));

        r = (lower_y_val                       + 359 * (v_val - 128) + 128) >> 8;
        g = (lower_y_val -  88 * (u_val - 128) + 183 * (v_val - 128) + 128) >> 8;
        b = (lower_y_val + 454 * (u_val - 128) +                       128) >> 8;

        rgbp[w * 3 + width * 3    ] = (uint8_t)(std::clamp<int32_t>(r, 0, 255));
        rgbp[w * 3 + width * 3 + 1] = (uint8_t)(std::clamp<int32_t>(g, 0, 255));
        rgbp[w * 3 + width * 3 + 2] = (uint8_t)(std::clamp<int32_t>(b, 0, 255));
      }
    }
  };

  int64_t height_2 = height / 2;

  int64_t num_threads = (int64_t)eigen_device->numThreads();
  int64_t block_size = (height_2 + num_threads - 1) / num_threads;
  int64_t need_num_threads = (height_2 + block_size - 1) / block_size;

  Eigen::Barrier barrier((unsigned int)(need_num_threads));

  for (int64_t i = 0; i < need_num_threads; ++i) {
    int64_t start_h = i * block_size * 2;
    int64_t end_h = std::min<int64_t>(start_h + block_size * 2, height);

    eigen_device->enqueue_with_barrier(
      &barrier,
      block,
      y,
      uv,
      rgb,
      height,
      width,
      start_h,
      end_h);
  }

  barrier.Wait();
}

//To convert Y'UV to RGB :
//matrix from :
//|R|   | 1    0     359 | | Y'      |
//|G| = | 1 - 88     183 | | U - 128 |
//|B|   | 1   454     0  | | V - 128 |
//in integer math :
//R = clamp(((Y)                  + 359*(V-128) + 128) >> 8)
//G = clamp(((Y) -  88 * (U-128)  + 183*(V-128) + 128) >> 8)
//B = clamp(((Y) + 454 * (U-128)                + 128) >> 8)
Tensor nv12_2_rgb_full(const Tensor &y, const Tensor &uv) {
  ARGUMENT_CHECK(3 == y.shape().ndims() && 1 == y.shape()[2], "y shape error");
  ARGUMENT_CHECK(3 == uv.shape().ndims() && 2 == uv.shape()[2], "uv shape error");
  ARGUMENT_CHECK(y.element_type() == uv.element_type(), "y uv element type must same");

  int64_t height = y.shape()[0];
  int64_t width = y.shape()[1];

  ARGUMENT_CHECK(height == 2 * uv.shape()[0] && width == 2 * uv.shape()[1], "y/uv shape error");

  auto rgb = Tensor::create({ height, width, 3 }, y.element_type());

  if (y.element_type().is<uint8_t>()) {
    nv12_2_rgb_full_uint8_impl(y.eigen_device().get(), y.data<uint8_t>(), uv.data<uint8_t>(), rgb.data<uint8_t>(), height, width);
  } else {
    RUNTIME_ERROR("element type:" << y.element_type().name() << " not support!");
  }

  return rgb;
}

}