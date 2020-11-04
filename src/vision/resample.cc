#include "vision/resample.h"

namespace dlfm::vision {

Tensor resize(const Tensor &img, std::vector<int64_t> size, std::string mode, bool align_corners) {
  if ("n" == mode || "nearest" == mode) {
    return resample_nearest(img, size);
  } else if ("b" == mode || "bilinear" == mode) {
    return resample_bilinear(img, size, align_corners);
  } else {
    RUNTIME_ERROR("not support mode:" << mode);
  }
}

inline float cal_origin_index(int64_t origin_size, int64_t up_size, int64_t up_idx, bool align_corners) {
  if (align_corners) {
    return 1.0 * up_idx * (origin_size - 1) / (up_size - 1);
  } else {
    float origin_idx = (up_idx + 0.5) * origin_size / up_size - 0.5;

    return origin_idx < 0 ? 0 : origin_idx;
  }
}

////////////////////////////////////////////////////////////////////////////////////////
// nearest
template <typename T>
void resample_nearest_block_impl(
  T *input,
  T *output,
  int64_t input_height,
  int64_t input_width,
  int64_t output_height,
  int64_t output_width,
  int64_t channel,
  int64_t start_index,
  int64_t end_index) {
  for (int64_t oy = start_index; oy < end_index; ++oy) {
    int64_t iy = round(cal_origin_index(input_height, output_height, oy, false));

    for (int64_t ox = 0; ox < output_width; ++ox) {
      int64_t ix = round(cal_origin_index(input_width, output_width, ox, false));

      T *in = input + iy * input_width * channel  + ix * channel;
      T *out = output + oy * output_height * channel + ox * channel;

      for (int64_t c = 0; c < channel; ++c) {
        out[c] = in[c];
      }
    }
  }
}

template <typename T>
void resample_nearest_impl(
  Eigen::ThreadPoolDevice *eigen_device,
  T *x,
  T *y,
  int64_t input_height,
  int64_t input_width,
  int64_t output_height,
  int64_t output_width,
  int64_t channel) {
  int64_t num_threads = (int64_t)eigen_device->numThreads();
  int64_t block_size = (output_height + num_threads - 1) / num_threads;
  int64_t need_num_threads = (output_height + block_size - 1) / block_size;

  Eigen::Barrier barrier((unsigned int)(need_num_threads));

  for (int64_t i = 0; i < need_num_threads; ++i) {
    int64_t start_index = i * block_size;
    int64_t end_index = std::min<int64_t>(start_index + block_size, output_height);

    eigen_device->enqueue_with_barrier(
      &barrier,
      &resample_nearest_block_impl<T>,
      x,
      y,
      input_height,
      input_width,
      output_height,
      output_width,
      channel,
      start_index,
      end_index);
  }

  barrier.Wait();
}

Tensor resample_nearest(const Tensor &img, std::vector<int64_t> size) {
  ARGUMENT_CHECK(3 == img.shape().ndims(), "img dimension must be 3");
  ARGUMENT_CHECK(size[0] > 0 && size[1] > 0, "size must > 0");

  int64_t input_height = img.shape()[0];
  int64_t input_width = img.shape()[1];

  int64_t channel = img.shape()[2];

  int64_t output_height = size[0];
  int64_t output_width = size[1];

  auto output = Tensor::create({ output_height , output_width , channel}, img.element_type());

  if (img.element_type().is<float>()) {
    resample_nearest_impl(
      img.eigen_device().get(),
      img.data<float>(),
      output.data<float>(),
      input_height,
      input_width,
      output_height,
      output_width,
      channel);
  } else if (img.element_type().is<uint8_t>()) {
    resample_nearest_impl(
      img.eigen_device().get(),
      img.data<uint8_t>(),
      output.data<uint8_t>(),
      input_height,
      input_width,
      output_height,
      output_width,
      channel);
  } else {
    RUNTIME_ERROR("element type:" << img.element_type().name() << " not support!");
  }

  return output;
}

////////////////////////////////////////////////////////////////////////////////////////
// bilinear

template <typename T>
void resample_bilinear_block_impl(
  T *input,
  T *output,
  int64_t input_height,
  int64_t input_width,
  int64_t output_height,
  int64_t output_width,
  int64_t channel,
  int64_t start_index,
  int64_t end_index,
  bool align_corners) {

  float t_lowest = float(std::numeric_limits<T>::lowest());
  float t_max = float((std::numeric_limits<T>::max)());

  for (int64_t o_y = start_index; o_y < end_index; ++o_y) {
    float i_y = cal_origin_index(input_height, output_height, o_y, align_corners);

    int64_t i_y_r = floor(i_y);
    int64_t i_y_p = i_y_r < (input_height - 1) ? 1 + i_y_r : i_y_r;

    float y_r_factor = i_y - i_y_r;
    float y_p_factor = 1.0 - y_r_factor;

    for (int64_t o_x = 0; o_x < output_width; ++o_x) {
      float i_x = cal_origin_index(input_width, output_width, o_x, align_corners);

      int64_t i_x_r = floor(i_x);
      int64_t i_x_p = i_x_r < (input_width - 1) ? 1 + i_x_r : i_x_r;

      float x_r_factor = i_x - i_x_r;
      float x_p_factor = 1.0 - x_r_factor;

      T *left_top = input + i_y_r * input_width * channel + i_x_r * channel;
      T *right_top = input + i_y_r * input_width * channel + i_x_p * channel;
      T *left_bottom = input + i_y_p * input_width * channel + i_x_r * channel;
      T *right_bottom = input + i_y_p * input_width * channel + i_x_p * channel;

      T *out = output + o_y * output_width * channel + o_x * channel;

      for (int64_t c = 0; c < channel; ++c) {
        float t = y_p_factor * (float(left_top[c]) * x_p_factor + float(right_top[c]) * x_r_factor) +
                  y_r_factor * (float(left_bottom[c]) * x_p_factor + float(right_bottom[c]) * x_r_factor);

         t = std::clamp<float>(t, t_lowest, t_max);

         out[c] = (T)t;
      }
    }
  }
}

template <typename T>
void resample_bilinear_impl(
  Eigen::ThreadPoolDevice *eigen_device,
  T *x,
  T *y,
  int64_t input_height,
  int64_t input_width,
  int64_t output_height,
  int64_t output_width,
  int64_t channel,
  bool align_corners) {
  int64_t num_threads = (int64_t)eigen_device->numThreads();
  int64_t block_size = (output_height + num_threads - 1) / num_threads;
  int64_t need_num_threads = (output_height + block_size - 1) / block_size;

  Eigen::Barrier barrier((unsigned int)(need_num_threads));

  for (int64_t i = 0; i < need_num_threads; ++i) {
    int64_t start_index = i * block_size;
    int64_t end_index = std::min<int64_t>(start_index + block_size, output_height);

    eigen_device->enqueue_with_barrier(
      &barrier,
      &resample_bilinear_block_impl<T>,
      x,
      y,
      input_height,
      input_width,
      output_height,
      output_width,
      channel,
      start_index,
      end_index,
      align_corners);
  }

  barrier.Wait();
}

Tensor resample_bilinear(const Tensor &img, std::vector<int64_t> size, bool align_corners) {
  ARGUMENT_CHECK(3 == img.shape().ndims(), "img dimension must be 3");
  ARGUMENT_CHECK(size[0] > 0 && size[1] > 0, "size must > 0");

  int64_t input_height = img.shape()[0];
  int64_t input_width = img.shape()[1];

  int64_t channel = img.shape()[2];

  int64_t output_height = size[0];
  int64_t output_width = size[1];

  auto output = Tensor::create({ output_height , output_width , channel }, img.element_type());

  if (img.element_type().is<float>()) {
    resample_bilinear_impl(
      img.eigen_device().get(),
      img.data<float>(),
      output.data<float>(),
      input_height,
      input_width,
      output_height,
      output_width,
      channel,
      align_corners);
  } else if (img.element_type().is<uint8_t>()) {
    resample_bilinear_impl(
      img.eigen_device().get(),
      img.data<uint8_t>(),
      output.data<uint8_t>(),
      input_height,
      input_width,
      output_height,
      output_width,
      channel,
      align_corners);
  } else {
    RUNTIME_ERROR("element type:" << img.element_type().name() << " not support!");
  }

  return output;
}


}