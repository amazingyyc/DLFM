#include "math/pad.h"

namespace dlfm {
namespace math {

template <size_t ndims, typename T>
void pad_impl(Eigen::ThreadPoolDevice *eigen_device, T *x, const Shape &xshape, T *y, const Shape &yshape, const std::vector<size_t> &paddings) {
  Eigen::array<Eigen::Index, ndims> xdims;
  Eigen::array<Eigen::Index, ndims> ydims;

  for (int i = 0; i < ndims; ++i) {
    xdims[i] = xshape[i];
    ydims[i] = yshape[i];
  }

  Eigen::array<std::pair<size_t, size_t>, ndims> e_paddings;

  for (int i = 0; i < ndims; ++i) {
    e_paddings[ndims - i - 1] = std::make_pair(paddings[2 * i], paddings[2 * i + 1]);
  }

  Eigen::TensorMap<Eigen::Tensor<T, ndims, Eigen::RowMajor>> xvec(x, xdims);
  Eigen::TensorMap<Eigen::Tensor<T, ndims, Eigen::RowMajor>> yvec(y, ydims);

  yvec.device(*eigen_device) = xvec.pad(e_paddings);
}

// not test
template <typename T>
void pad_impl_v2(
  std::shared_ptr<Device> device,
  T *x, const Shape &xshape,
  T *y, const Shape &yshape,
  const std::vector<size_t> &paddings) {
  // assume y shape: [d0, d1,..., dn] become [row, col]
  // row = d0 * d1 *...* dn-1, col = dn
  auto block = [] (
    std::shared_ptr<Device> device,
    T *x, const Shape &xshape,
    T *y, const Shape &yshape,
    const std::vector<size_t> &paddings,
    int64_t start_row,
    int64_t end_row) {
    int64_t ndims = xshape.ndims();

    int64_t last_pad_left = paddings[0];
    int64_t last_pad_right = paddings[1];

    int64_t xcol = xshape[-1];
    int64_t ycol = yshape[-1];

    for (int64_t r = start_row; r < end_row; ++r) {
      int64_t offset = r * ycol;
      int64_t y_offset = offset;
      int64_t x_offset = 0;

      bool valid = true;

      for (int64_t i = 0; i + 1 < ndims; ++i) {
        int64_t y_idx = offset / yshape.stride(i);
        int64_t x_idx = y_idx - (int64_t)(paddings[2 * (ndims - i - 1)]);

        if (x_idx < 0 || x_idx >= xshape[i]) {
          valid = false;
          break;
        }

        x_offset += x_idx * xshape.stride(i);

        offset %= yshape.stride(i);
      }

      if (valid) {
        device->zero(y + y_offset, sizeof(T) * last_pad_left);
        device->memcpy(y + y_offset + last_pad_left, x + x_offset, sizeof(T) * xcol);
        device->zero(y + y_offset + last_pad_left + xcol, sizeof(T) * last_pad_right);
      } else {
        device->zero(y + y_offset, sizeof(T) * ycol);
      }
    }
  };

  int64_t ycol = yshape[-1];
  int64_t yrow = yshape.size() / ycol;

  int64_t num_threads = (int64_t)device->eigen_device()->numThreads();
  int64_t block_size = (yrow + num_threads - 1) / num_threads;

  num_threads = (yrow + block_size - 1) / block_size;

  Eigen::Barrier barrier((unsigned int)(num_threads));

  for (int64_t i = 0; i < num_threads; ++i) {
    int64_t start_row = i * block_size;
    int64_t end_row = (std::min)(start_row + block_size, yrow);

    device->eigen_device()->enqueue_with_barrier(
      &barrier,
      block,
      device,
      x,
      xshape,
      y,
      yshape,
      paddings,
      start_row,
      end_row);
  }

  barrier.Wait();
}

void pad(const Tensor &x, Tensor &y, std::vector<size_t> paddings) {
  ARGUMENT_CHECK(x.shape().ndims() == y.shape().ndims() && 
                2 * x.shape().ndims() == paddings.size(),
                "pad shape error");

  if (x.element_type().is<float>()) {
    if (1 == x.shape().ndims()) {
      pad_impl<1, float>(x.eigen_device().get(), x.data<float>(), x.shape(), y.data<float>(), y.shape(), paddings);
    } else if (2 == x.shape().ndims()) {
      pad_impl<2, float>(x.eigen_device().get(), x.data<float>(), x.shape(), y.data<float>(), y.shape(), paddings);
    } else if (3 == x.shape().ndims()) {
      pad_impl<3, float>(x.eigen_device().get(), x.data<float>(), x.shape(), y.data<float>(), y.shape(), paddings);
    } else if (4 == x.shape().ndims()) {
      pad_impl<4, float>(x.eigen_device().get(), x.data<float>(), x.shape(), y.data<float>(), y.shape(), paddings);
    } else {
      RUNTIME_ERROR("the rank of input is error");
    }
  } else {
    RUNTIME_ERROR("element type:" << x.element_type().name() << " not support!");
  }
}

}
}