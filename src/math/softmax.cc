#include "math/softmax.h"
#include "math/neon_mathfun.h"

namespace dlfm::math {

void softmax_3d_f32_vertical_impl(Eigen::ThreadPoolDevice *eigen_device, float *x, float *y, int64_t row, int64_t col) {
  auto block = [](float *x, float *y, int64_t row, int64_t col) {
    float *xptr = x;
    float *yptr = y;

    float max_v = -(std::numeric_limits<float>::max)();

    for (int64_t r = 0; r < row; ++r) {
      max_v = std::max<float>(max_v, xptr[0]);

      xptr += col;
    }

    xptr = x;
    yptr = y;

    for (int64_t r = 0; r < row; ++r) {
      yptr[0] = std::exp(xptr[0] - max_v);

      xptr += col;
      yptr += col;
    }

    yptr = y;
    float e_sum_v = 0;

    for (int64_t r = 0; r < row; ++r) {
      e_sum_v += yptr[0];

      yptr += col;
    }

    yptr = y;
    for (int64_t r = 0; r < row; ++r) {
      yptr[0] = yptr[0] / e_sum_v;

      yptr += col;
    }
  };

  Eigen::Barrier barrier((unsigned int)(col));

  for (int64_t idx = 0; idx < col; ++idx) {
    eigen_device->enqueue_with_barrier(&barrier, block, x + idx, y + idx, row, col);
  }

  barrier.Wait();
}

// x [do, d1, d2]
// mean [d0, d1, 1]
// y [d0, d1, 1]
void softmax_3d_f32_horizontal_impl(Eigen::ThreadPoolDevice *eigen_device, float *x, float *y, int64_t row, int64_t col) {
  auto block = [](float *x, float *y, int64_t row, int64_t col) {
    int64_t limit = col / 4 * 4;

    float max_val = -(std::numeric_limits<float>::max)();
    float e_sum_val = 0;

    {
      int64_t c = 0;

#if defined(__ARM_NEON__)
      float32x4_t max_val_v = vdupq_n_f32(-std::numeric_limits<float>::max());

      for (; c < limit; c += 4) {
        float32x4_t x_val = vld1q_f32(x + c);

        max_val_v = vmaxq_f32(x_val, max_val_v);
      }

      max_val = std::max(max_val_v[0], max_val);
      max_val = std::max(max_val_v[1], max_val);
      max_val = std::max(max_val_v[2], max_val);
      max_val = std::max(max_val_v[3], max_val);
#endif

      for (; c < col; ++c) {
        max_val = (std::max)(x[c], max_val);
      }
    }

    {
      int64_t c = 0;

#if defined(__ARM_NEON__)
      float32x4_t max_val_v = vdupq_n_f32(max_val);

      for (; c < limit; c += 4) {
        float32x4_t x_val = vld1q_f32(x + c);
        vst1q_f32(y + c, neon::exp_ps(vsubq_f32(x_val, max_val_v)));
      }
#endif

      for (; c < col; ++c) {
        y[c] = std::exp(x[c] - max_val);
      }
    }

    {
      int64_t c = 0;

#if defined(__ARM_NEON__)
      float32x4_t e_sum_val_v = vdupq_n_f32(0);

      for (; c < limit; c += 4) {
        float32x4_t y_val = vld1q_f32(y + c);
        e_sum_val_v = vaddq_f32(e_sum_val_v, y_val);
      }

      e_sum_val += e_sum_val_v[0] + e_sum_val_v[1] + e_sum_val_v[2] + e_sum_val_v[3];
#endif

      for (; c < col; ++c) {
        e_sum_val += y[c];
      }
    }

    {
      int64_t c = 0;

#if defined(__ARM_NEON__)
      float32x4_t e_sum_val_v = vdupq_n_f32(e_sum_val);

      for (; c < limit; c += 4) {
        float32x4_t y_val = vld1q_f32(y + c);

        vst1q_f32(y + c, vdivq_f32(y_val, e_sum_val_v));
      }
#endif

      for (; c < col; ++c) {
        y[c] = y[c] / e_sum_val;
      }
    }
  };

  Eigen::Barrier barrier((unsigned int)(row));

  for (int64_t idx = 0; idx < row; ++idx) {
    eigen_device->enqueue_with_barrier(&barrier, block, x + idx * col, y + idx * col, row, col);
  }

  barrier.Wait();
}

void softmax_3d_f32_middle_impl(Eigen::ThreadPoolDevice *eigen_device, float *x, float *y, int64_t d0, int64_t d1, int64_t d2) {
  auto block = [](float *x, float *y, int64_t d1, int64_t d2) {
    float *xptr = x;
    float *yptr = y;

    float max_v = -(std::numeric_limits<float>::max)();
    float e_sum_v = 0;

    for (int64_t d = 0; d < d1; ++d) {
      max_v = std::max<float>(max_v, xptr[0]);

      xptr += d2;
    }

    xptr = x;
    yptr = y;

    for (int64_t d = 0; d < d1; ++d) {
      yptr[0] = std::exp(xptr[0] - max_v);

      xptr += d2;
      yptr += d2;
    }

    yptr = y;
    for (int64_t d = 0; d < d1; ++d) {
      e_sum_v += yptr[0];

      yptr += d2;
    }

    yptr = y;
    for (int64_t d = 0; d < d1; ++d) {
      yptr[0] = yptr[0] / e_sum_v;

      yptr += d2;
    }
  };

  Eigen::Barrier barrier((unsigned int)(d0 * d2));

  for (int64_t idx = 0; idx < d0 * d2; ++idx) {
    int64_t idx0 = idx / d2;
    int64_t idx2 = idx % d2;

    eigen_device->enqueue_with_barrier(&barrier, block, x + idx0 * d1 * d2 + idx2, y + idx0 * d1 * d2 + idx2, d1, d2);
  }

  barrier.Wait();
}


void softmax(const Tensor &x, Tensor &y, int64_t axis) {
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

  if (1 == d2) {
    softmax_3d_f32_horizontal_impl(x.eigen_device().get(), x.data<float>(), y.data<float>(), d0, d1 * d2);
  } else if (1 == d0) {
    softmax_3d_f32_vertical_impl(x.eigen_device().get(), x.data<float>(), y.data<float>(), d0 * d1, d2);
  } else {
    softmax_3d_f32_middle_impl(x.eigen_device().get(), x.data<float>(), y.data<float>(), d0, d1, d2);
  }
}

}
















