#include "math/var.h"

namespace dlfm {
namespace math {

// x [do, d1, d2]
// mean [1, d1, d2]
// y [1, d1, d2]
void var_3d_f32_vertical_impl(Eigen::ThreadPoolDevice *eigen_device, float *x, float *mean, float *y, float scale, int64_t row, int64_t col) {
  auto block = [](float *x, float *mean, float *y, float scale, int64_t row, int64_t col) {
    float variance = 0;

    for (int64_t r = 0; r < row; ++r) {
      float diff = x[0] - mean[0];
      variance += diff * diff;

      x += col;
    }

    y[0] = variance * scale;
  };

  Eigen::Barrier barrier((unsigned int)(col));

  for (int64_t idx = 0; idx < col; ++idx) {
    eigen_device->enqueue_with_barrier(&barrier, block, x + idx, mean + idx, y + idx, scale, row, col);
  }

  barrier.Wait();
}

// x [do, d1, d2]
// mean [d0, 1, d2]
// y [d0, 1, d2]
void var_3d_f32_middle_impl(Eigen::ThreadPoolDevice *eigen_device, float *x, float *mean, float *y, float scale, int64_t d0, int64_t d1, int64_t d2) {
  auto block = [](float *x, float *mean, float *y, float scale, int64_t d1, int64_t d2) {
    float variance = 0;

    for (int64_t d = 0; d < d1; ++d) {
      float diff = x[0] - mean[0];
      variance += diff * diff;

      x += d2;
    }

    y[0] = variance * scale;
  };

  Eigen::Barrier barrier((unsigned int)(d0 * d2));

  for (int64_t idx = 0; idx < d0 * d2; ++idx) {
    int64_t idx0 = idx / d2;
    int64_t idx2 = idx % d2;

    eigen_device->enqueue_with_barrier(&barrier, block, x + idx0 * d1 * d2 + idx2, mean + idx0 * d2 + idx2, y + idx0 * d2 + idx2, scale, d1, d2);
  }

  barrier.Wait();
}

// x [do, d1, d2]
// mean [d0, d1, 1]
// y [d0, d1, 1]
void var_3d_f32_horizontal_impl(Eigen::ThreadPoolDevice *eigen_device, float *x, float *mean, float *y, float scale, int64_t row, int64_t col) {
  auto block = [](float *x, float *mean, float *y, float scale, int64_t row, int64_t col) {
    float variance = 0;

    int64_t c = 0;
    int64_t limit = col / 4 * 4;

#if defined(__ARM_NEON__)
    float32x4_t meanv = vdupq_n_f32(mean[0]);
    float32x4_t sumv = vdupq_n_f32(0);

    for (; c < limit; c += 4) {
      float32x4_t diff = vsubq_f32(vld1q_f32(x + c), meanv);

      sumv = vaddq_f32(sumv, vmulq_f32(diff, diff));
    }

    variance += sumv[0] + sumv[1] + sumv[2] + sumv[3];
#endif

    for (; c < col; ++c) {
      float diff = x[c] - mean[0];
      variance += diff * diff;
    }

    y[0] = variance * scale;
  };

  Eigen::Barrier barrier((unsigned int)(row));

  for (int64_t idx = 0; idx < row; ++idx) {
    eigen_device->enqueue_with_barrier(&barrier, block, x + idx * col, mean + idx, y + idx, scale, row, col);
  }

  barrier.Wait();
}

void var(const Tensor &x, const Tensor &mean, int64_t axis, bool unbiased, Tensor &y) {
  ARGUMENT_CHECK(x.element_type().is<float>(), "var only support float");

  auto ndims = x.ndims();

  if (axis < 0) {
    ndims += ndims;
  }

  int64_t d0 = 1;
  int64_t d1 = 1;
  int64_t d2 = 1;

  d1 = x.shape()[axis];

  for (int64_t i = 0; i < axis; ++i) {
    d0 *= x.shape()[i];
  }

  for (int64_t i = axis + 1; i < ndims; ++i) {
    d2 *= x.shape()[i];
  }

  float scale = 1.0 / float(d1);

  if (unbiased) {
    scale = 1.0 / float(d1 - 1);
  }

  if (1 == d2) {
    var_3d_f32_horizontal_impl(x.eigen_device().get(), x.data<float>(), mean.data<float>(), y.data<float>(), scale, d0, d1 * d2);
  } else if (1 == d0) {
    var_3d_f32_vertical_impl(x.eigen_device().get(), x.data<float>(), mean.data<float>(), y.data<float>(), scale, d0 * d1, d2);
  } else {
    var_3d_f32_middle_impl(x.eigen_device().get(), x.data<float>(), mean.data<float>(), y.data<float>(), scale, d0, d1, d2);
  }
}

}
}