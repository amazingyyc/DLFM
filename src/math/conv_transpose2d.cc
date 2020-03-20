#include "math/conv_transpose2d.h"

namespace dlfm {
namespace math {

template <typename T>
void conv_transpose2d_impl(Eigen::ThreadPoolDevice *eigen_device,
                           T *in_ptr,
                           T *weight_ptr,
                           T *bias_ptr,
                           T *out_ptr,
                           int64_t batch,
                           int64_t in_channel,
                           int64_t in_height,
                           int64_t in_width,
                           int64_t out_channel,
                           int64_t out_height,
                           int64_t out_width,
                           std::vector<size_t> kernel_size,
                           std::vector<size_t> stride,
                           std::vector<size_t> padding,
                           std::vector<size_t> out_padding) {
    Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>> in(in_ptr, batch, in_channel, in_height, in_width);
    Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>> weight(weight_ptr, in_channel, out_channel, (int64_t)kernel_size[0], (int64_t)kernel_size[1]);
    Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>> bias(bias_ptr, 1, out_channel, 1, 1);
    Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>> out(out_ptr, batch, out_channel, out_height, out_width);

    // extract_image_patches support input dimension [N, columns, rows, channel]
    //    extract_image_patches(const Index patch_rows, const Index patch_cols,
    //    const Index row_stride, const Index col_stride,
    //    const Index in_row_stride, const Index in_col_stride,
    //    const Index row_inflate_stride, const Index col_inflate_stride,

    //    const Index padding_top, const Index padding_bottom,
    //    const Index padding_left,const Index padding_right,
    //    const Scalar padding_value) const {

    // shuffle -> [batch, h, w, c]
    // extract_image_patches -> [batch, out_h * out_w, k_h, k_w, in_channel]
    // reshape -> [batch, out_h * out_w, k_h*k_w*in_channel]
    // weight [in_channel, out_channle, k_h, k_w] reshape -> [out_channel, k_h * k_w * in_channel]
    Eigen::Index  padding_top    = kernel_size[1] - padding[1] - 1;
    Eigen::Index  padding_bottom = kernel_size[1] - padding[1] - 1 + out_padding[1];
    Eigen::Index  padding_left   = kernel_size[0] - padding[0] - 1;
    Eigen::Index  padding_right  = kernel_size[0] - padding[0] - 1 + out_padding[0];

    Eigen::Index kernel_size_0 = (Eigen::Index)kernel_size[0];
    Eigen::Index kernel_size_1 = (Eigen::Index)kernel_size[1];

    Eigen::array<Eigen::Index, 4> in_shuffle = {0, 2, 3, 1};
    Eigen::array<Eigen::Index, 2> extract_reshape = {batch * out_height * out_width, kernel_size_0 * kernel_size_1 * in_channel};
    Eigen::array<bool, 4> weight_reverse = {false, false, true, true};
    Eigen::array<Eigen::Index, 4> weight_shuffle = {1, 2, 3, 0};
    Eigen::array<Eigen::Index, 2> weight_reshape = {out_channel, kernel_size_0 * kernel_size_1 * in_channel};
    Eigen::array<Eigen::Index, 4> out_reshape = { batch, out_height, out_width, out_channel};
    Eigen::array<Eigen::Index, 4> out_shuffle = {0, 3, 1, 2};
    Eigen::array<Eigen::Index, 4> bias_cast = { batch,  1, out_height, out_width};

    Eigen::array<Eigen::IndexPair<int>, 1> contract_dims = { Eigen::IndexPair<int>(1, 1) };

    out.device(*eigen_device) = in.shuffle(in_shuffle).extract_image_patches(kernel_size_1, kernel_size_0,
                                   1, 1,
                                   1, 1,
                                   stride[1], stride[0],
                                   padding_top, padding_bottom,
                                   padding_left, padding_right, 0)
            .reshape(extract_reshape)
            .contract(weight.reverse(weight_reverse).shuffle(weight_shuffle).reshape(weight_reshape), contract_dims).reshape(out_reshape).shuffle(out_shuffle) + bias.broadcast(bias_cast);
}

void conv_transpose2d(const Tensor &input,
                      const Tensor &weight,
                      const Tensor &bias,
                      Tensor &output,
                      std::vector<size_t> kernel_size,
                      std::vector<size_t> stride,
                      std::vector<size_t> padding,
                      std::vector<size_t> out_padding) {
    if (input.element_type().is<float>()) {
        conv_transpose2d_impl<float>(input.eigen_device().get(),
                                     input.data<float>(),
                                     weight.data<float>(),
                                     bias.data<float>(),
                                     output.data<float>(),
                                     input.shape()[0],
                                     input.shape()[1],
                                     input.shape()[2],
                                     input.shape()[3],
                                     output.shape()[1],
                                     output.shape()[2],
                                     output.shape()[3],
                                     kernel_size,
                                     stride,
                                     padding,
                                     out_padding);
    } else {
        RUNTIME_ERROR("element type:" << input.element_type().name() << " nor support!");
    }
}

}
}