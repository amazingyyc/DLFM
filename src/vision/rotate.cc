#include "vision/rotate.h"

namespace dlfm::vision {

template <typename T>
void rotate_right_90_block_impl(
  std::shared_ptr<Device> device,
  T *x,
  T *y,
  int64_t x_height,
  int64_t x_width,
  int64_t channel,
  int64_t start,
  int64_t end) {
  int64_t bytes_count = sizeof(T) * channel;

  for (int64_t i = start; i < end; ++i) {
    for (int64_t j = 0; j < x_height; ++j) {
      device->memcpy(y + i * x_height * channel + j * channel, x + (x_height - 1 - j) * x_width * channel + i * channel, bytes_count);
    }
  }
}

template <typename T>
void rotate_right_90_impl(
  std::shared_ptr<Device> device,
  T *x,
  T *y,
  int64_t x_height,
  int64_t x_width,
  int64_t channel) {
  int64_t num_threads = (int64_t)device->eigen_device()->numThreads();
  int64_t block_size = (x_width + num_threads - 1) / num_threads;
  int64_t need_num_threads = (x_width + block_size - 1) / block_size;

  Eigen::Barrier barrier((unsigned int)(need_num_threads));

  for (int64_t i = 0; i < need_num_threads; ++i) {
    int64_t start_index = i * block_size;
    int64_t end_index = std::min<int64_t>(start_index + block_size, x_width);

    device->eigen_device()->enqueue_with_barrier(
      &barrier,
      &rotate_right_90_block_impl<T>,
      device,
      x,
      y,
      x_height,
      x_width,
      channel,
      start_index,
      end_index);
  }

  barrier.Wait();
}

// rotate the image right_90 degree
Tensor rotate_right_90(const Tensor &x) {
  ARGUMENT_CHECK(3 == x.shape().ndims(), "x ndims must be 3.");

  int64_t x_height = x.shape()[0];
  int64_t x_width = x.shape()[1];
  int64_t channel = x.shape()[2];

  auto y = Tensor::create({ x_width , x_height , channel }, x.element_type());

  if (x.element_type().is<float>()) {
    rotate_right_90_impl(
      x.device(),
      x.data<float>(),
      y.data<float>(),
      x_height,
      x_width,
      channel
    );
  } else if (x.element_type().is<uint8_t>()) {
    rotate_right_90_impl(
      x.device(),
      x.data<uint8_t>(),
      y.data<uint8_t>(),
      x_height,
      x_width,
      channel
    );
  } else {
    RUNTIME_ERROR("element type:" << x.element_type().name() << " not support!");
  }

  return y;
}

//------------------------------------------------------
// rotate
inline bool valid_idx(int64_t start, int64_t end, int64_t idx) {
  return (idx >= start) && (idx < end);
}

template <typename T>
void rotate_block_impl(
  std::shared_ptr<Device> device,
  T *x,
  T *y,
  T *pad,
  int64_t height,
  int64_t width,
  int64_t channel,
  float m00,
  float m01,
  float m02,
  float m10,
  float m11,
  float m12,
  int64_t start,
  int64_t end) {
  float t_lowest = float(std::numeric_limits<T>::lowest());
  float t_max = float(std::numeric_limits<T>::max());

  for (int64_t yh = start; yh < end; ++yh) {
    for (int64_t yw = 0; yw < width; ++yw) {
      T *out = y + yh * width * channel + yw * channel;

      float xw = m00 * yw + m01 * yh + m02;
      float xh = m10 * yw + m11 * yh + m12;

      int64_t xw_l = floor(xw);
      int64_t xw_r = xw_l + 1;

      int64_t xh_t = floor(xh);
      int64_t xh_b = xh_t + 1;

      if (!valid_idx(0, width, xw_l) &&
          !valid_idx(0, width, xw_r) &&
          !valid_idx(0, height, xh_t) &&
          !valid_idx(0, height, xh_b)) {
        device->memcpy(out, pad, sizeof(T) * channel);
        continue;
      }

      float xw_l_factor = xw - xw_l;
      float xw_r_factor = 1.0f - xw_l_factor;

      float xh_t_factor = xh - xh_t;
      float xh_b_factor = 1.0f - xh_t_factor;

      T *left_top = (valid_idx(0, height, xh_t) && valid_idx(0, width, xw_l)) ? (x + xh_t * width * channel + xw_l * channel) : pad;
      T *right_top = (valid_idx(0, height, xh_t) && valid_idx(0, width, xw_r)) ? (x + xh_t * width * channel + xw_r * channel) : pad;
      T *left_bottom = (valid_idx(0, height, xh_b) && valid_idx(0, width, xw_l)) ? (x + xh_b * width * channel + xw_l * channel) : pad;
      T *right_bottom = (valid_idx(0, height, xh_b) && valid_idx(0, width, xw_r)) ? (x + xh_b * width * channel + xw_r * channel) : pad;

      for (int64_t c = 0; c < channel; ++c) {
        float t = xh_b_factor * (float(left_top[c]) * xw_r_factor + float(right_top[c]) * xw_l_factor) +
                  xh_t_factor * (float(left_bottom[c]) * xw_r_factor + float(right_bottom[c]) * xw_l_factor);

        t = std::clamp<float>(t, t_lowest, t_max);

        out[c] = (T)t;
      }
    }
  }
}

template <typename T>
void rotate_impl(
  std::shared_ptr<Device> device,
  T *x,
  T *y,
  T *pad,
  int64_t height,
  int64_t width,
  int64_t channel,
  float m00,
  float m01,
  float m02,
  float m10,
  float m11,
  float m12) {
  int64_t num_threads = (int64_t)device->eigen_device()->numThreads();
  int64_t block_size = (height + num_threads - 1) / num_threads;
  int64_t need_num_threads = (height + block_size - 1) / block_size;

  Eigen::Barrier barrier((unsigned int)(need_num_threads));

  for (int64_t i = 0; i < need_num_threads; ++i) {
    int64_t start_index = i * block_size;
    int64_t end_index = std::min<int64_t>(start_index + block_size, height);

    device->eigen_device()->enqueue_with_barrier(
      &barrier,
      &rotate_block_impl<T>,
      device,
      x,
      y,
      pad,
      height,
      width,
      channel,
      m00,
      m01,
      m02,
      m10,
      m11,
      m12,
      start_index,
      end_index);
  }

  barrier.Wait();
}

Tensor rotate(const Tensor &x, float angle, const Tensor &pad) {
  ARGUMENT_CHECK(3 == x.shape().ndims(), "img dimension must be 3");
  ARGUMENT_CHECK(x.shape()[2] == pad.shape()[0], "x pad channel must same");

  int64_t height  = x.shape()[0];
  int64_t width   = x.shape()[1];
  int64_t channel = x.shape()[2];

  auto y = Tensor::create(x.shape(), x.element_type());

  float cos_a = cos(angle);
  float sin_a = sin(angle);

  float m00 = cos_a;
  float m01 = sin_a;
  float m02 = -(float(width) / 2.0f) * cos_a - (float(height) / 2.0f) * sin_a + float(width) / 2.0f;

  float m10 = -sin_a;
  float m11 = cos_a;
  float m12 = (float(width) / 2.0f) * sin_a - (float(height) / 2.0f) * cos_a + float(height) / 2.0f;

  if (x.element_type().is<float>()) {
    rotate_impl(
      x.device(),
      x.data<float>(),
      y.data<float>(),
      pad.data<float>(),
      height,
      width,
      channel,
      m00,
      m01,
      m02,
      m10,
      m11,
      m12);
  } else if (x.element_type().is<uint8_t>()) {
    rotate_impl(
      x.device(),
      x.data<uint8_t>(),
      y.data<uint8_t>(),
      pad.data<uint8_t>(),
      height,
      width,
      channel,
      m00,
      m01,
      m02,
      m10,
      m11,
      m12);
    } else {
      RUNTIME_ERROR("element type:" << x.element_type().name() << " not support!");
    }

  return y;
}

Tensor affine_transform(const Tensor &x, const std::vector<float> &matrix, const Tensor &pad) {
  ARGUMENT_CHECK(3 == x.shape().ndims(), "img dimension must be 3");
  ARGUMENT_CHECK(x.shape()[2] == pad.shape()[0], "x pad channel must same");

  int64_t height  = x.shape()[0];
  int64_t width   = x.shape()[1];
  int64_t channel = x.shape()[2];

  auto y = Tensor::create(x.shape(), x.element_type());

  float m00 = matrix[0];
  float m01 = matrix[1];
  float m02 = matrix[2];

  float m10 = matrix[3];
  float m11 = matrix[4];
  float m12 = matrix[5];

  if (x.element_type().is<float>()) {
    rotate_impl(
            x.device(),
            x.data<float>(),
            y.data<float>(),
            pad.data<float>(),
            height,
            width,
            channel,
            m00,
            m01,
            m02,
            m10,
            m11,
            m12);
  } else if (x.element_type().is<uint8_t>()) {
    rotate_impl(
            x.device(),
            x.data<uint8_t>(),
            y.data<uint8_t>(),
            pad.data<uint8_t>(),
            height,
            width,
            channel,
            m00,
            m01,
            m02,
            m10,
            m11,
            m12);
  } else {
    RUNTIME_ERROR("element type:" << x.element_type().name() << " not support!");
  }

  return y;
}

}