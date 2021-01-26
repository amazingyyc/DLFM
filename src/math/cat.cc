#include "math/cat.h"

namespace dlfm {
namespace math {

template <size_t ndims, typename T>
void cat_impl(Eigen::ThreadPoolDevice *eigen_device, T *x, const Shape &xshape, T *y, const Shape &yshape, T *z, const Shape &zshape, int64_t axis) {
  Eigen::array<Eigen::Index, ndims> xdims;
  Eigen::array<Eigen::Index, ndims> ydims;
  Eigen::array<Eigen::Index, ndims> zdims;

  for (int i = 0; i < ndims; ++i) {
    xdims[i] = xshape[i];
    ydims[i] = yshape[i];
    zdims[i] = zshape[i];
  }

  Eigen::TensorMap<Eigen::Tensor<T, ndims, Eigen::RowMajor>> xvec(x, xdims);
  Eigen::TensorMap<Eigen::Tensor<T, ndims, Eigen::RowMajor>> yvec(y, ydims);
  Eigen::TensorMap<Eigen::Tensor<T, ndims, Eigen::RowMajor>> zvec(z, zdims);

  zvec.device(*eigen_device) = xvec.concatenate(yvec, axis);
}

void cat(const Tensor &x, const Tensor &y, Tensor &z, int64_t axis) {
  if (x.element_type().is<float>()) {
    if (1 == x.shape().ndims()) {
      cat_impl<1, float>(x.eigen_device().get(), x.data<float>(), x.shape(), y.data<float>(), y.shape(), z.data<float>(), z.shape(), axis);
    } else if (2 == x.shape().ndims()) {
      cat_impl<2, float>(x.eigen_device().get(), x.data<float>(), x.shape(), y.data<float>(), y.shape(), z.data<float>(), z.shape(), axis);
    } else if (3 == x.shape().ndims()) {
      cat_impl<3, float>(x.eigen_device().get(), x.data<float>(), x.shape(), y.data<float>(), y.shape(), z.data<float>(), z.shape(), axis);
    } else if (4 == x.shape().ndims()) {
      cat_impl<4, float>(x.eigen_device().get(), x.data<float>(), x.shape(), y.data<float>(), y.shape(), z.data<float>(), z.shape(), axis);
    } else {
      RUNTIME_ERROR("the rank of input is error");
    }
  } else if (x.element_type().is<uint8_t>()) {
    if (1 == x.shape().ndims()) {
      cat_impl<1, uint8_t>(x.eigen_device().get(), x.data<uint8_t>(), x.shape(), y.data<uint8_t>(), y.shape(), z.data<uint8_t>(), z.shape(), axis);
    } else if (2 == x.shape().ndims()) {
      cat_impl<2, uint8_t>(x.eigen_device().get(), x.data<uint8_t>(), x.shape(), y.data<uint8_t>(), y.shape(), z.data<uint8_t>(), z.shape(), axis);
    } else if (3 == x.shape().ndims()) {
      cat_impl<3, uint8_t>(x.eigen_device().get(), x.data<uint8_t>(), x.shape(), y.data<uint8_t>(), y.shape(), z.data<uint8_t>(), z.shape(), axis);
    } else if (4 == x.shape().ndims()) {
      cat_impl<4, uint8_t>(x.eigen_device().get(), x.data<uint8_t>(), x.shape(), y.data<uint8_t>(), y.shape(), z.data<uint8_t>(), z.shape(), axis);
    } else {
      RUNTIME_ERROR("the rank of input is error");
    }
  } else {
    RUNTIME_ERROR("element type:" << x.element_type().name() << " not support!");
  }
}

// x's shape: [m, k1, n] + [m, k2, n] ... [m, kt, n] => [m, sum(k1...kt), n]
template <typename T>
void cat_v2_impl(
  std::shared_ptr<Device> device,
  const std::vector<T*> &x,
  T *y,
  int64_t m,
  const std::vector<int64_t> &k,
  int64_t n,
  int64_t k_total) {
  auto block = [](
    std::shared_ptr<Device> device,
    const std::vector<T*> &x,
    T *y,
    int64_t m_start,
    int64_t m_end,
    const std::vector<int64_t> &k,
    int64_t n,
    int64_t k_total) {
    for (int64_t m_idx = m_start; m_idx < m_end; ++m_idx) {
      int64_t y_offset = m_idx * k_total * n;

      for (int64_t i = 0; i < x.size(); ++i) {
        int64_t cur_x_offset = m_idx * k[i] * n;
        int64_t cur_x_lenght = k[i] * n;

        device->memcpy(y + y_offset, x[i] + cur_x_offset, cur_x_lenght * sizeof(T));

        y_offset += cur_x_lenght;
      }
    }
  };

  int64_t num_threads = (int64_t)device->eigen_device()->numThreads();
  int64_t block_size = (m + num_threads - 1) / num_threads;

  num_threads = (m + block_size - 1) / block_size;

  Eigen::Barrier barrier((unsigned int)(num_threads));

  for (int64_t i = 0; i < num_threads; ++i) {
    int64_t m_start = i * block_size;
    int64_t m_end = (std::min)(m_start + block_size, m);

    device->eigen_device()->enqueue_with_barrier(
      &barrier,
      block,
      device,
      x,
      y,
      m_start,
      m_end,
      k,
      n,
      k_total);
  }

  barrier.Wait();
}

void cat_v2(const std::vector<Tensor> &x, Tensor &y, int64_t axis) {
  int64_t ndims = y.ndims();

  if (axis < 0) {
    axis += ndims;
  }

  ARGUMENT_CHECK(0 <= axis && axis < ndims, "axis out of range");

  int64_t m = 1;
  std::vector<int64_t> k;
  int64_t k_total = 0;
  int64_t n = 1;

  for (int64_t i = 0; i < ndims; ++i) {
    if (i != axis) {
      for (auto &item : x) {
        ARGUMENT_CHECK(y.shape()[i] == item.shape()[i], "shape error");
      }

      if (i < axis) {
        m *= y.shape()[i];
      } else {
        n *= y.shape()[i];
      }
    } else {
      for (auto &item : x) {
        k_total += item.shape()[i];
        k.emplace_back(item.shape()[i]);
      }
    }
  }

  ARGUMENT_CHECK(k_total == y.shape()[axis], "shape error");

  if (y.element_type().is<float>()) {
    std::vector<float*> xptrs;
    float *yptr = y.data<float>();

    for (auto &item : x) {
      xptrs.emplace_back(item.data<float>());
    }

    cat_v2_impl<float>(
      y.device(),
      xptrs,
      yptr,
      m,
      k,
      n,
      k_total);
  } else if (y.element_type().is<uint8_t>()) {
    std::vector<uint8_t*> xptrs;
    uint8_t *yptr = y.data<uint8_t>();

    for (auto &item : x) {
      xptrs.emplace_back(item.data<uint8_t>());
    }

    cat_v2_impl<uint8_t>(
      y.device(),
      xptrs,
      yptr,
      m,
      k,
      n,
      k_total);
  } else {
    RUNTIME_ERROR("element type:" << y.element_type().name() << " not support!");
  }
}

}
}