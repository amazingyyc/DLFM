#include "math/unary_cwise.h"

namespace dlfm {
namespace math {

template <typename T>
void assign_impl(Eigen::ThreadPoolDevice *eigen_device, T *x, T *y, int64_t n) {
  Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> xvec(x, n);
  Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> yvec(y, n);

  yvec.device(*eigen_device) = xvec;
}

void assign(const Tensor &x, Tensor &y) {
  if (x.element_type().is<float>()) {
    assign_impl<float>(x.eigen_device().get(), x.data<float>(), y.data<float>(), x.size());
  } else {
    RUNTIME_ERROR("element type:" << x.element_type().name() << " nor support!");
  }
}

template <typename T>
void add_impl(Eigen::ThreadPoolDevice *eigen_device, T *x, T value, T *y, int64_t n) {
  Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> xvec(x, n);
  Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> yvec(y, n);

  yvec.device(*eigen_device) = xvec + value;
}

void add(const Tensor &x, float value, Tensor &y) {
  if (x.element_type().is<float>()) {
    add_impl<float>(x.eigen_device().get(), x.data<float>(), value, y.data<float>(), x.size());
  } else {
    RUNTIME_ERROR("element type:" << x.element_type().name() << " nor support!");
  }
}

template <typename T>
void sub_impl(Eigen::ThreadPoolDevice *eigen_device, T *x, T value, T *y, int64_t n) {
  Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> xvec(x, n);
  Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> yvec(y, n);

  yvec.device(*eigen_device) = xvec - value;
}

void sub(const Tensor &x, float value, Tensor &y) {
  if (x.element_type().is<float>()) {
    sub_impl<float>(x.eigen_device().get(), x.data<float>(), value, y.data<float>(), x.size());
  } else {
    RUNTIME_ERROR("element type:" << x.element_type().name() << " nor support!");
  }
}

template <typename T>
void multiply_impl(Eigen::ThreadPoolDevice *eigen_device, T *x, T value, T *y, int64_t n) {
  Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> xvec(x, n);
  Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> yvec(y, n);

  yvec.device(*eigen_device) = xvec * value;
}

void multiply(const Tensor &x, float value, Tensor &y) {
  if (x.element_type().is<float>()) {
    multiply_impl<float>(x.eigen_device().get(), x.data<float>(), value, y.data<float>(), x.size());
  } else {
    RUNTIME_ERROR("element type:" << x.element_type().name() << " nor support!");
  }
}

void divide(const Tensor &x, float value, Tensor &y) {
  multiply(x, 1.0 / value, y);
}

void add(float value, const Tensor &x, Tensor &y) {
  return add(x, value, y);
}

void sub(float value, const Tensor &x, Tensor &y) {
  auto block = [] (float value, float *x, float *y, int64_t n) {
    int64_t idx = 0;
    int64_t limit = n / 4 * 4;

#if defined(__ARM_NEON__)
    float32x4_t value_v = vdupq_n_f32(value);
    
    for (; idx < limit; idx += 4) {
      float32x4_t xv = vld1q_f32(x + idx);
      vst1q_f32(y + idx, vsubq_f32(value_v, xv));
    }
#endif

    for (; idx < n; ++idx) {
      y[idx] = value - x[idx];
    }
  };

  int64_t n = x.size();
  float *xptr = x.data<float>();
  float *yptr = y.data<float>();
  auto eigen_device = x.eigen_device();

  int64_t num_threads = (int64_t)(eigen_device->numThreads());
  int64_t block_size  = (n + num_threads - 1) / num_threads;

  num_threads = (n + block_size - 1) / block_size;

  Eigen::Barrier barrier((unsigned int)(num_threads));

  for (int64_t i = 0; i < num_threads; ++i) {
    int64_t start = i * block_size;
    int64_t real_block_size = (std::min)(block_size, n - start);

    eigen_device->enqueue_with_barrier(&barrier, block, value, xptr + start, yptr + start, real_block_size);
  }

  barrier.Wait();
}

void multiply(float value, const Tensor &x, Tensor &y) {
  return multiply(x, value, y);
}

void divide(float value, const Tensor &x, Tensor &y) {
  auto block = [] (float value, float *x, float *y, int64_t n) {
    int64_t idx = 0;
    int64_t limit = n / 4 * 4;

#if defined(__ARM_NEON__)
    float32x4_t value_v = vdupq_n_f32(value);
    
    for (; idx < limit; idx += 4) {
      float32x4_t xv = vld1q_f32(x + idx);
      vst1q_f32(y + idx, vdivq_f32(value_v, xv));
    }
#endif

    for (; idx < n; ++idx) {
      y[idx] = value / x[idx];
    }
  };

  int64_t n = x.size();
  float *xptr = x.data<float>();
  float *yptr = y.data<float>();
  auto eigen_device = x.eigen_device();

  int64_t num_threads = (int64_t)(eigen_device->numThreads());
  int64_t block_size  = (n + num_threads - 1) / num_threads;

  num_threads = (n + block_size - 1) / block_size;

  Eigen::Barrier barrier((unsigned int)(num_threads));

  for (int64_t i = 0; i < num_threads; ++i) {
    int64_t start = i * block_size;
    int64_t real_block_size = (std::min)(block_size, n - start);

    eigen_device->enqueue_with_barrier(&barrier, block, value, xptr + start, yptr + start, real_block_size);
  }

  barrier.Wait();
}

}
}