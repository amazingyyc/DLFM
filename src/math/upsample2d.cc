#include "math/upsample2d.h"

namespace dlfm {
namespace math {

float cal_origin_index(int64_t origin_size, int64_t up_size, int64_t up_idx, bool align_corners) {
  if (align_corners) {
    return 1.0 * up_idx * (origin_size - 1) / (up_size - 1);
  } else {
    float origin_idx = (up_idx + 0.5) * origin_size / up_size - 0.5;

    return origin_idx < 0 ? 0 : origin_idx;
  }
}

template <typename T>
void upsample_nearest2d_block_impl(
    T *input,
    T *output,
    int64_t start_channel,
    int64_t end_channel,
    int64_t input_height,
    int64_t input_width,
    int64_t output_height,
    int64_t output_width) {
  for (int64_t o_y = 0; o_y < output_height; ++o_y) {
    for (int64_t o_x = 0; o_x < output_width; ++o_x) {
      int64_t i_y = round(cal_origin_index(input_height, output_height, o_y, false));
      int64_t i_x = round(cal_origin_index(input_width, output_width, o_x, false));

      T *in  = input  + start_channel * input_height * input_width + i_y * input_width + i_x;
      T *out = output + start_channel * output_height * output_width + o_y * output_width + o_x;

      for (int64_t c = start_channel; c < end_channel; ++c) {
        out[0] = in[0];

        in += input_height * input_width;
        out += output_height * output_width;
      }
    }
  }
}

template <typename T>
void upsample_nearest2d_impl(
              Eigen::ThreadPoolDevice *eigen_device,
              T *x, const Shape &x_shape,
              T *y, const Shape &y_shape) {
  int64_t channel      = x_shape[0] * x_shape[1];
  int64_t input_height = x_shape[2];
  int64_t input_width  = x_shape[3];

  int64_t output_height = y_shape[2];
  int64_t output_width  = y_shape[3];

  int64_t num_threads = (int64_t)eigen_device->numThreads();
  int64_t block_size  = (channel + num_threads - 1) / num_threads;
  int64_t need_num_threads = (channel + block_size - 1) / block_size;

  Eigen::Barrier barrier((unsigned int)(need_num_threads));

  for (int64_t i = 0; i < need_num_threads; ++i) {
    int start_channel = i * block_size;
    int end_channel = std::min<int64_t>(start_channel + block_size, channel);

    eigen_device->enqueue_with_barrier(
        &barrier,
        &upsample_nearest2d_block_impl<T>,
        x,
        y,
        start_channel,
        end_channel,
        input_height,
        input_width,
        output_height,
        output_width);
  }

  barrier.Wait();
}

void upsample_nearest2d(const Tensor &x, Tensor &y) {
  if (x.element_type().is<float>()) {
    upsample_nearest2d_impl<float>(x.eigen_device().get(), x.data<float>(), x.shape(), y.data<float>(), y.shape());
  } else {
    RUNTIME_ERROR("element type:" << x.element_type().name() << " nor support!");
  }
}

// bilinear
template <typename T>
void upsample_bilinear2d_block_impl(
    T *input,
    T *output,
    int64_t start_channel,
    int64_t end_channel,
    int64_t input_height,
    int64_t input_width,
    int64_t output_height,
    int64_t output_width,
    bool align_corners) {
  for (int64_t o_y = 0; o_y < output_height; ++o_y) {
    float i_y = cal_origin_index(input_height, output_height, o_y, align_corners);

    int64_t i_y_r = floor(i_y);
    int64_t i_y_p = i_y_r < (input_height - 1) ? 1 : 0;

    float y_r_factor = i_y - i_y_r;
    float y_p_factor = 1.0 - y_r_factor;

    for (int64_t o_x = 0; o_x < output_width; ++o_x) {
      float i_x = cal_origin_index(input_width, output_width, o_x, align_corners);

      int64_t i_x_r = floor(i_x);
      int64_t i_x_p = i_x_r < (input_width - 1) ? 1 : 0;

      float x_r_factor = i_x - i_x_r;
      float x_p_factor = 1.0 - x_r_factor;

      T *in  = input  + start_channel * input_height * input_width  + i_y_r * input_width + i_x_r;
      T *out = output + start_channel * output_height * output_width + o_y * output_width + o_x;

      for (int64_t c = start_channel; c < end_channel; ++c) {
        out[0] = y_p_factor * (in[0] * x_p_factor + in[i_x_p] * x_r_factor) +
                 y_r_factor * (in[i_y_p * input_width] * x_p_factor + in[i_y_p * input_width + i_x_p] * x_r_factor);

        in  += input_height * input_width;
        out += output_height * output_width;
      }
    }
  }
}

template <typename T>
void upsample_bilinear2d_impl(
    Eigen::ThreadPoolDevice *eigen_device,
    T *x, const Shape &x_shape,
    T *y, const Shape &y_shape,
    bool align_corners) {
  int64_t channel      = x_shape[0] * x_shape[1];
  int64_t input_height = x_shape[2];
  int64_t input_width  = x_shape[3];

  int64_t output_height = y_shape[2];
  int64_t output_width  = y_shape[3];

  int64_t num_threads = (int64_t)eigen_device->numThreads();

  int64_t block_size  = (channel + num_threads - 1) / num_threads;
  int64_t need_num_threads = (channel + block_size - 1) / block_size;

  Eigen::Barrier barrier((unsigned int)(need_num_threads));

  for (int64_t i = 0; i < need_num_threads; ++i) {
    int start_channel = i * block_size;
    int end_channel = std::min<int64_t>(start_channel + block_size, channel);

    eigen_device->enqueue_with_barrier(
        &barrier,
        &upsample_bilinear2d_block_impl<T>,
        x,
        y,
        start_channel,
        end_channel,
        input_height,
        input_width,
        output_height,
        output_width,
        align_corners
    );
  }

  barrier.Wait();
}

void upsample_bilinear2d(const Tensor &x, Tensor &y, bool align_corners) {
  if (x.element_type().is<float>()) {
    upsample_bilinear2d_impl<float>(x.eigen_device().get(), x.data<float>(), x.shape(), y.data<float>(), y.shape(), align_corners);
  } else {
    RUNTIME_ERROR("element type:" << x.element_type().name() << " nor support!");
  }
}








}
}