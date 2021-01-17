#ifdef HAS_NNPACK
#include "nnpack.h"
#endif

#include "math/conv2d.h"

namespace dlfm::math {

#ifdef HAS_NNPACK
// stride is 1x1 and group is 1
void conv2d_f32_stride1x1_group1_impl(
  const Tensor &input,
  const Tensor &weight,
  const Tensor &bias,
  Tensor &output,
  const std::vector<size_t> &padding) {
  float *input_ptr  = input.data<float>();
  float *weight_ptr = weight.data<float>();
  float *bias_ptr   = bias.data<float>();
  float *output_ptr = output.data<float>();

  int64_t batch = input.shape()[0];

  int64_t input_channel = input.shape()[-3];
  int64_t input_height  = input.shape()[-2];
  int64_t input_width   = input.shape()[-1];

  int64_t output_channel = output.shape()[-3];
  int64_t output_height  = output.shape()[-2];
  int64_t output_width   = output.shape()[-1];

  int64_t kernel_height = weight.shape()[2];
  int64_t kernel_width = weight.shape()[3];

  struct nnp_size input_size;
  input_size.width  = (size_t)input_width;
  input_size.height = (size_t)input_height;

  struct nnp_padding input_padding;
  input_padding.top = padding[0];
  input_padding.right = padding[1];
  input_padding.bottom = padding[0];
  input_padding.left = padding[1];

  struct nnp_size kernel_size;
  kernel_size.width = (size_t)kernel_width;
  kernel_size.height = (size_t)kernel_height;

  // for save memory, when the input image is too big use gemm
  auto convolution_algorithm = nnp_convolution_algorithm_auto;
  // if (input_height * input_width * output_channel >= 512 * 512 * 64) {
  //   convolution_algorithm = nnp_convolution_algorithm_implicit_gemm;
  // }

  auto nnp_status = nnp_convolution_output(
    convolution_algorithm,
    batch,
    input_channel,
    output_channel,
    input_size,
    input_padding,
    kernel_size,
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    output.nnpack_threadpool(),
    nullptr);

  ARGUMENT_CHECK(nnp_status == nnp_status_success, "nnp_convolution_output error:" << nnp_status);
}

void conv2d_f32_batch1_group1_impl(
  const Tensor &input,
  const Tensor &weight,
  const Tensor &bias,
  Tensor &output,
  const std::vector<size_t> &stride,
  const std::vector<size_t> &padding) {
  float *input_ptr  = input.data<float>();
  float *weight_ptr = weight.data<float>();
  float *bias_ptr   = bias.data<float>();
  float *output_ptr = output.data<float>();

  int64_t input_channel = input.shape()[-3];
  int64_t input_height  = input.shape()[-2];
  int64_t input_width   = input.shape()[-1];

  int64_t output_channel = output.shape()[-3];
  int64_t output_height  = output.shape()[-2];
  int64_t output_width   = output.shape()[-1];

  int64_t kernel_height = weight.shape()[2];
  int64_t kernel_width = weight.shape()[3];

  struct nnp_size input_size;
  input_size.width  = (size_t)input_width;
  input_size.height = (size_t)input_height;

  struct nnp_padding input_padding;
  input_padding.top = padding[0];
  input_padding.right = padding[1];
  input_padding.bottom = padding[0];
  input_padding.left = padding[1];

  struct nnp_size kernel_size;
  kernel_size.width = (size_t)kernel_width;
  kernel_size.height = (size_t)kernel_height;

  struct nnp_size output_subsampling;
  output_subsampling.width = stride[1];
  output_subsampling.height = stride[0];

  // for save memory, when the input image is too big use gemm
  auto convolution_algorithm = nnp_convolution_algorithm_auto;
  // if (input_height * input_width * output_channel >= 512 * 512 * 64) {
  //   convolution_algorithm = nnp_convolution_algorithm_implicit_gemm;
  // }

  auto nnp_status = nnp_convolution_inference(
    convolution_algorithm,
    nnp_convolution_transform_strategy_block_based,
    input_channel,
    output_channel,
    input_size,
    input_padding,
    kernel_size,
    output_subsampling,
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    output.nnpack_threadpool(),
    nullptr);

  ARGUMENT_CHECK(nnp_status == nnp_status_success, "nnp_convolution_inference error:" << nnp_status);
}

void conv2d_f32_impl(
  const Tensor &input,
  const Tensor &weight,
  const Tensor &bias,
  Tensor &output,
  const std::vector<size_t> &stride,
  const std::vector<size_t> &padding,
  const size_t groups) {
  float *input_ptr  = input.data<float>();
  float *weight_ptr = weight.data<float>();
  float *bias_ptr   = bias.data<float>();
  float *output_ptr = output.data<float>();

  int64_t batch = input.shape()[0];

  int64_t input_channel = input.shape()[-3];
  int64_t input_height  = input.shape()[-2];
  int64_t input_width   = input.shape()[-1];

  int64_t output_channel = output.shape()[-3];
  int64_t output_height  = output.shape()[-2];
  int64_t output_width   = output.shape()[-1];

  int64_t kernel_height = weight.shape()[2];
  int64_t kernel_width = weight.shape()[3];

  struct nnp_size input_size;
  input_size.width  = (size_t)input_width;
  input_size.height = (size_t)input_height;

  struct nnp_size output_size;
  output_size.width  = (size_t)output_width;
  output_size.height = (size_t)output_height;

  struct nnp_padding input_padding;
  input_padding.top = padding[0];
  input_padding.right = padding[1];
  input_padding.bottom = padding[0];
  input_padding.left = padding[1];

  struct nnp_size kernel_size;
  kernel_size.width = (size_t)kernel_width;
  kernel_size.height = (size_t)kernel_height;

  struct nnp_size output_subsampling;
  output_subsampling.width = stride[1];
  output_subsampling.height = stride[0];

  int64_t input_group_channel = input_channel / groups;
  int64_t output_group_channel = output_channel / groups;

  // for save memory, when the input image is too big use gemm
  auto convolution_algorithm = nnp_convolution_algorithm_auto;
  // if (input_height * input_width * input_group_channel >= 512 * 512 * 64) {
  //   convolution_algorithm = nnp_convolution_algorithm_implicit_gemm;
  // }

  int64_t batch_x_groups = batch * groups;

  int64_t num_threads = (int64_t)output.eigen_device()->numThreads();
  int64_t block_size = (batch_x_groups + num_threads - 1) / num_threads;

  num_threads = (batch_x_groups + block_size - 1) / block_size;

  Eigen::Barrier barrier((unsigned int)(num_threads));
  Eigen::ThreadPoolDevice *eigen_device = output.eigen_device().get();

  auto block = [] (
    enum nnp_convolution_algorithm algorithm,
    enum nnp_convolution_transform_strategy transform_strategy,
    size_t input_channels,
    size_t output_channels,
    struct nnp_size input_size,
    struct nnp_padding input_padding,
    struct nnp_size kernel_size,
    struct nnp_size output_subsampling,
    const float input[],
    const float kernel[],
    const float bias[],
    float output[],
    pthreadpool_t threadpool,
    struct nnp_profile* profile) {
      auto nnp_status = nnp_convolution_inference(
        algorithm,
        transform_strategy,
        input_channels,
        output_channels,
        input_size,
        input_padding,
        kernel_size,
        output_subsampling,
        input,
        kernel,
        bias,
        output,
        nullptr,
        nullptr);

      ARGUMENT_CHECK(nnp_status == nnp_status_success, "nnp_convolution_inference error:" << nnp_status);
  };

  auto block_v2 = [](
    enum nnp_convolution_algorithm algorithm,
    enum nnp_convolution_transform_strategy transform_strategy,
    int64_t batch,
    int64_t groups,
    size_t input_group_channel,
    size_t output_group_channel,
    struct nnp_size input_size,
    struct nnp_padding input_padding,
    struct nnp_size kernel_size,
    struct nnp_size output_subsampling,
    struct nnp_size output_size,
    const float *input,
    const float *weight,
    const float *bias,
    float *output,
    int64_t start,
    int64_t end) {
    for (int64_t idx = start; idx < end; ++idx) {
      int64_t b_idx = idx / groups;
      int64_t g_idx = idx % groups;

      auto nnp_status = nnp_convolution_inference(
        algorithm,
        transform_strategy,
        input_group_channel,
        output_group_channel,
        input_size,
        input_padding,
        kernel_size,
        output_subsampling,
        input + idx * input_group_channel * input_size.height * input_size.width,
        weight + g_idx * output_group_channel * input_group_channel * kernel_size.height * kernel_size.width,
        bias + g_idx * output_group_channel,
        output + idx * output_group_channel * output_size.height * output_size.width,
        nullptr,
        nullptr);

      ARGUMENT_CHECK(nnp_status == nnp_status_success, "nnp_convolution_inference error:" << nnp_status);
    }
  };

  for (int64_t i = 0; i < num_threads; ++i) {
    int64_t start = i * block_size;
    int64_t end = (std::min)(start + block_size, batch_x_groups);

    eigen_device->enqueue_with_barrier(
      &barrier,
      block_v2,
      convolution_algorithm,
      nnp_convolution_transform_strategy_block_based,
      batch,
      groups,
      input_group_channel,
      output_group_channel,
      input_size,
      input_padding,
      kernel_size,
      output_subsampling,
      output_size,
      input_ptr,
      weight_ptr,
      bias_ptr,
      output_ptr,
      start,
      end
    );
  }

  barrier.Wait();

  // for (int64_t b_idx = 0; b_idx < batch; ++b_idx) {
  //   for (int64_t g_idx = 0; g_idx < groups; ++g_idx) {
  //     auto idx = b_idx * groups + g_idx;

  //     eigen_device->enqueue_with_barrier(
  //       &barrier,
  //       block,
  //       convolution_algorithm,
  //       nnp_convolution_transform_strategy_block_based,
  //       input_group_channel,
  //       output_group_channel,
  //       input_size,
  //       input_padding,
  //       kernel_size,
  //       output_subsampling,
  //       input_ptr + idx * input_group_channel * input_height * input_width,
  //       weight_ptr + g_idx * output_group_channel * input_group_channel * kernel_height * kernel_width,
  //       bias_ptr + g_idx * output_group_channel,
  //       output_ptr + idx * output_group_channel * output_height * output_width,
  //       nullptr,
  //       nullptr);
  //   }
  // }
}

#endif

/************************************************************************/
/*nnpack not support dilation, implement a special conv2d with dilation.*/
/************************************************************************/
inline float vector_multiply(float *v1, float *v2, int64_t length) {
  int64_t limit = length / 4 * 4;

  float sum = 0;
  int64_t c = 0;

#if defined(__ARM_NEON__)
  float32x4_t sumv = vdupq_n_f32(0);

  for (; c < limit; c += 4) {
    float32x4_t v1v = vld1q_f32(v1 + c);
    float32x4_t v2v = vld1q_f32(v2 + c);

    sumv = vaddq_f32(sumv, vmulq_f32(v1v, v2v));
  }

  sum += sumv[0] + sumv[1] + sumv[2] + sumv[3];
#endif

  for (; c < length; ++c) {
    sum += v1[c] * v2[c];
  }

  return sum;
}

void conv2d_with_dilation_f32_block_impl(
  std::shared_ptr<Device> device,
  float *in,
  float *weight,
  float *bias,
  float *out,
  int64_t in_channel,
  int64_t in_height,
  int64_t in_width,
  int64_t out_channel,
  int64_t out_height,
  int64_t out_width,
  int64_t kernel_height,
  int64_t kernel_width,
  int64_t stride_y,
  int64_t stride_x,
  int64_t padding_top,
  int64_t padding_left,
  int64_t dilation_y,
  int64_t dilation_x
  ) {
  // Use tmp memory to store input.
  float *mid = (float*)device->allocate(in_channel * kernel_height * kernel_width * sizeof(float));

  for (int64_t oc = 0; oc < out_channel; ++oc) {
    for (int64_t oy = 0; oy < out_height; ++oy) {
      for (int64_t ox = 0; ox < out_width; ++ox) {
        // Copy input to in_t;
        for (int64_t ic = 0; ic < in_channel; ++ic) {
          for (int64_t wy = 0, iy = oy * stride_y - padding_top; wy < kernel_height; ++wy, iy += dilation_y) {
            for (int64_t wx = 0, ix = ox * stride_x - padding_left; wx < kernel_width; ++wx, ix += dilation_x) {
              if (0 <= iy && iy < in_height && 0 <= ix && ix < in_width) {
                mid[(ic * kernel_height + wy) * kernel_width + wx] = in[(ic * in_height + iy) * in_width + ix];
              } else {
                mid[(ic * kernel_height + wy) * kernel_width + wx] = 0;
              }
            }
          }
        }

        // Calculate vector multiply.
        out[(oc * out_height + oy) * out_width + ox] =
          bias[oc] + vector_multiply(mid, weight + oc * in_channel * kernel_height * kernel_width,
                                     in_channel * kernel_height * kernel_width);
      }
    }
  }

  device->deallocate(mid);
}

void conv2d_with_dilation_f32_impl(
  const Tensor &in_t,
  const Tensor &weight_t,
  const Tensor &bias_t,
  Tensor &out_t,
  const std::vector<size_t> &stride,
  const std::vector<size_t> &padding,
  const std::vector<size_t> &dilation,
  const size_t groups) {
  float *in     = in_t.data<float>();
  float *weight = weight_t.data<float>();
  float *bias = bias_t.data<float>();
  float *out  = out_t.data<float>();

  int64_t batch = in_t.shape()[0];

  int64_t in_channel = in_t.shape()[-3];
  int64_t in_height  = in_t.shape()[-2];
  int64_t in_width   = in_t.shape()[-1];

  int64_t out_channel = out_t.shape()[-3];
  int64_t out_height  = out_t.shape()[-2];
  int64_t out_width   = out_t.shape()[-1];

  int64_t kernel_height = weight_t.shape()[2];
  int64_t kernel_width  = weight_t.shape()[3];

  int64_t stride_y = stride[0];
  int64_t stride_x = stride[1];
  int64_t padding_top  = padding[0];
  int64_t padding_left = padding[1];
  int64_t dilation_y = dilation[0];
  int64_t dilation_x = dilation[1];

  int64_t in_group_channel = in_channel / groups;
  int64_t out_group_channel = out_channel / groups;

  auto device = out_t.device();
  auto eigen_device = out_t.eigen_device();
  Eigen::Barrier barrier((unsigned int)(batch * groups * out_group_channel));

  for (int64_t bidx = 0; bidx < batch; ++bidx) {
    for (int64_t gidx = 0; gidx < groups; ++gidx) {
      for (int64_t oidx = 0; oidx < out_group_channel; ++oidx) {
        eigen_device->enqueue_with_barrier(
          &barrier,
          &conv2d_with_dilation_f32_block_impl,
          device,
          in + (bidx * groups + gidx) * in_group_channel * in_height * in_width,
          weight + (gidx * out_group_channel + oidx) * in_group_channel * kernel_height * kernel_width,
          bias + gidx * out_group_channel + oidx,
          out + ((bidx * groups + gidx) * out_group_channel + oidx) * out_height * out_width,
          in_group_channel,
          in_height,
          in_width,
          1,
          out_height,
          out_width,
          kernel_height,
          kernel_width,
          stride_y,
          stride_x,
          padding_top,
          padding_left,
          dilation_y,
          dilation_x);
      }
    }
  }

  barrier.Wait();
}

void conv2d(
  const Tensor &input,
  const Tensor &weight,
  const Tensor &bias,
  Tensor &output,
  const std::vector<size_t> &stride,
  const std::vector<size_t> &padding,
  const std::vector<size_t> &dilation,
  size_t groups) {
  ARGUMENT_CHECK(input.element_type().is<float>(), "conv2d only support float");

  if (1 != dilation[0] || 1 != dilation[1]) {
    conv2d_with_dilation_f32_impl(
      input,
      weight,
      bias,
      output,
      stride,
      padding,
      dilation,
      groups);
  } else {
#ifdef HAS_NNPACK
    if (1 == input.shape()[0] && 1 == groups) {
      conv2d_f32_batch1_group1_impl(
        input,
        weight,
        bias,
        output,
        stride,
        padding);
    } else if (1 == stride[0] && 1 == stride[1] && 1 == groups) {
      conv2d_f32_stride1x1_group1_impl(
        input,
        weight,
        bias,
        output,
        padding);
    } else {
      conv2d_f32_impl(
        input,
        weight,
        bias,
        output,
        stride,
        padding,
        groups);
  }
#else
    RUNTIME_ERROR("please build with nnpack");
#endif
  }
}



}
