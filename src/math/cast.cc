#include "math/cast.h"

namespace dlfm {
namespace math {

template <typename From, typename To>
void cast_block_impl(From *x, To *y, int64_t n) {
  int64_t limit = n / 4 * 4;
  int64_t l = 0;

  for (; l < limit; l += 4) {
    y[l] = (To)(x[l]);
    y[l + 1] = (To)(x[l + 1]);
    y[l + 2] = (To)(x[l + 2]);
    y[l + 3] = (To)(x[l + 3]);
  }

  for (; l < n; ++l) {
    y[l] = (To)(x[l]);
  }
}

template <typename From, typename To>
void cast_impl(Eigen::ThreadPoolDevice *eigen_device, From *x, To *y, int64_t n) {
  int64_t num_threads = (int64_t)eigen_device->numThreads();
  int64_t block_size = (n + num_threads - 1) / num_threads;

  num_threads = (n + block_size - 1) / block_size;

  Eigen::Barrier barrier((unsigned int)(num_threads));

  for (int64_t i = 0; i < num_threads; ++i) {
    int64_t start = i * block_size;
    int64_t real_block_size = (std::min)(block_size, n - start);

    eigen_device->enqueue_with_barrier(&barrier, &cast_block_impl<From, To>, x + start, y + start, real_block_size);
  }

  barrier.Wait();
}

uint32_t convert_mantissa(uint32_t i) {
  uint32_t m = i << 13;
  uint32_t e = 0;

  while (!(m & 0x00800000)) {
      e -= 0x00800000;
      m <<= 1;
  }

  m &= ~(0x00800000);
  e += 0x38800000;

  return m | e;
}

template <>
void cast_impl<float16, float>(Eigen::ThreadPoolDevice *eigen_device, float16 *x, float *y, int64_t n) {
  uint32_t exponent_table[64];
  uint32_t offset_table[64];
  uint32_t mantissa_table[2048];

  exponent_table[0]  = 0;
  exponent_table[32] = 0x80000000;
  exponent_table[31] = 0x47800000;
  exponent_table[63] = 0xC7800000;

  for (uint32_t i = 1; i <= 30; ++i) {
      exponent_table[i] = i << 23;
  }

  for (uint32_t i = 33; i <= 62; ++i) {
      exponent_table[i] = 0x80000000 + ((i - 32) << 23);
  }

  for (uint32_t i = 0; i < 64; ++i) {
      offset_table[i] = 1024;
  }

  offset_table[0]  = 0;
  offset_table[32] = 0;

  mantissa_table[0] = 0;

  for (int i = 1; i < 1024; ++i) {
      mantissa_table[i] = convert_mantissa(i);
  }

  for (int i = 1024; i < 2048; ++i) {
      mantissa_table[i] = 0x38000000 + ((i - 1024) << 13);
  }

  auto block = [] (uint16_t *from, uint32_t *to, int64_t n, uint32_t *exponent_table, uint32_t *offset_table, uint32_t *mantissa_table) {
    int64_t limit = n / 4 * 4;
    int64_t i = 0;

    for (; i < limit; i += 4) {
      to[i]   = mantissa_table[offset_table[from[i]   >> 10] + (from[i]   & 0x3ff)] + exponent_table[from[i]   >> 10];
      to[i+1] = mantissa_table[offset_table[from[i+1] >> 10] + (from[i+1] & 0x3ff)] + exponent_table[from[i+1] >> 10];
      to[i+2] = mantissa_table[offset_table[from[i+2] >> 10] + (from[i+2] & 0x3ff)] + exponent_table[from[i+2] >> 10];
      to[i+3] = mantissa_table[offset_table[from[i+3] >> 10] + (from[i+3] & 0x3ff)] + exponent_table[from[i+3] >> 10];
    }

    for (; i < n; ++i) {
        to[i] = mantissa_table[offset_table[from[i] >> 10] + (from[i] & 0x3ff)] + exponent_table[from[i] >> 10];
    }
  };

  int64_t num_threads = (int64_t)eigen_device->numThreads();
  int64_t block_size = (n + num_threads - 1) / num_threads;

  num_threads = (n + block_size - 1) / block_size;

  Eigen::Barrier barrier((unsigned int)(num_threads));

  for (int64_t i = 0; i < num_threads; ++i) {
    int64_t start = i * block_size;
    int64_t real_block_size = (std::min)(block_size, n - start);

    eigen_device->enqueue_with_barrier(
      &barrier,
      block,
      (uint16_t*)(x + start),
      (uint32_t*)(y + start),
      real_block_size,
      (uint32_t *)exponent_table,
      (uint32_t *)offset_table,
      (uint32_t *)mantissa_table);
  }

  barrier.Wait();
}

void cast(const Tensor &x, Tensor &y) {
  auto xtype = x.element_type();
  auto ytype = y.element_type();

  if (xtype.is<float>() && ytype.is<uint8_t>()) {
    cast_impl<float, uint8_t>(x.eigen_device().get(), x.data<float>(), y.data<uint8_t>(), x.size());
  } else if (xtype.is<uint8_t>() && ytype.is<float>()) {
    cast_impl<uint8_t, float>(x.eigen_device().get(), x.data<uint8_t>(), y.data<float>(), x.size());
  } else if (xtype.is<float16>() && ytype.is<float>()) {
    cast_impl<float16, float>(x.eigen_device().get(), x.data<float16>(), y.data<float>(), x.size());
  } else if (xtype.is<int64_t>() && ytype.is<float>()) {
    cast_impl<int64_t, float>(x.eigen_device().get(), x.data<int64_t>(), y.data<float>(), x.size());
  } else {
    RUNTIME_ERROR("cast from:" << xtype.name() << " to:" << ytype.name() << " not support.");
  }
}

}
}