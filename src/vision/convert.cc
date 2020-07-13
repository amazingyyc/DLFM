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

}