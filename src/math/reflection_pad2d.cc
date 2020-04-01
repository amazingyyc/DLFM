#include "math/reflection_pad2d.h"

namespace dlfm {
namespace math {

template <typename T>
void reflection_pad2d_block_impl(T *input,
                            T *output,
                            int64_t start_channel,
                            int64_t end_channel,
                            int64_t input_height,
                            int64_t input_width,
                            int64_t output_height,
                            int64_t output_width,
                            int64_t pad_top,
                            int64_t pad_bottom,
                            int64_t pad_left,
                            int64_t pad_right) {
  for (int64_t oy = 0; oy < output_height; ++oy) {
    int64_t iy = oy - pad_top;

    if (oy < pad_top) {
      iy = pad_top - oy;
    } else if (oy >= pad_top + input_height) {
      iy = 2 * input_height - oy + pad_top -2;
    }

    for (int64_t ox = 0; ox < output_width; ++ox) {
      int64_t ix = ox - pad_left;

      if (ox < pad_left) {
        ix = pad_left - ox;
      } else if (ox >= pad_left + input_width) {
        ix = 2 * input_width - ox + pad_left - 2;
      }

      T *in  = input  + start_channel * input_height  * input_width  + iy * input_width   + ix;
      T *out = output + start_channel * output_height * output_width + oy * output_height + ox;

      for (int64_t c = start_channel; c < end_channel; ++c) {
        out[0] = in[0];

        in  += input_height * input_width;
        out += output_height * output_width;
      }
    }
  }
}

template <typename T>
void reflection_pad2d_impl(Eigen::ThreadPoolDevice *eigen_device,
                      T *x,
                      const Shape &x_shape,
                      T *y,
                      const Shape &y_shape,
                      std::vector<size_t> pad) {
  int64_t channel = x_shape[0] * x_shape[1];
  int64_t input_height = x_shape[2];
  int64_t input_width  = x_shape[3];

  int64_t output_height = y_shape[2];
  int64_t output_width  = y_shape[3];

  int64_t pad_left   = pad[0];
  int64_t pad_right  = pad[1];
  int64_t pad_top    = pad[2];
  int64_t pad_bottom = pad[3];

  int64_t num_threads = (int64_t)eigen_device->numThreads();
  int64_t block_size = (channel + num_threads - 1) / num_threads;
  int64_t need_num_threads = (channel + block_size - 1) / block_size;

  Eigen::Barrier barrier((unsigned int)(need_num_threads));

  auto block = [&barrier](T *intput,
                          T *output,
                          int64_t start_channel,
                          int64_t end_channel,
                          int64_t input_height,
                          int64_t input_width,
                          int64_t output_height,
                          int64_t output_width,
                          int64_t pad_top,
                          int64_t pad_bottom,
                          int64_t pad_left,
                          int64_t pad_right) {
    reflection_pad2d_block_impl(
      intput,
      output,
      start_channel,
      end_channel,
      input_height,
      input_width,
      output_height,
      output_width,
      pad_top,
      pad_bottom,
      pad_left,
      pad_right);

    barrier.Notify();
  };

  for (int64_t i = 0; i < need_num_threads; ++i) {
    int start_channel = i * block_size;
    int end_channel = std::min<int64_t>(start_channel + block_size, channel);

    eigen_device->enqueueNoNotification(
      block,
      x,
      y,
      start_channel,
      end_channel,
      input_height,
      input_width,
      output_height,
      output_width,
      pad_top,
      pad_bottom,
      pad_left,
      pad_right);
  }

  barrier.Wait();
}

void reflection_pad2d(const Tensor &x, Tensor &y, std::vector<size_t> pad) {
  if (x.element_type().is<float>()) {
    reflection_pad2d_impl<float>(x.eigen_device().get(), x.data<float>(), x.shape(), y.data<float>(), y.shape(), pad);
  } else {
    RUNTIME_ERROR("element type:" << x.element_type().name() << " nor support!");
  }
}


}
}