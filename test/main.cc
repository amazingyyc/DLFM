#include <iostream>

#include "pad_test.h"
#include "cast_test.h"
#include "cat_test.h"
#include "transpose_test.h"
#include "conv2d_test.h"
#include "max_pooling2d_test.h"
#include "conv_transpose2d_test.h"
#include "upsample2d_test.h"
#include "matmul_test.h"
#include "unet_test.h"
#include "mean_test.h"

using namespace dlfm::test;

int main() {
  // cast_test();
  // pad_test();
  // cat_test();
  // conv2d_test();
  // conv_transpose2d_test();
  // max_pooling2d_test();
  // upsample2d_test();
  // matmul_test();
  // var_test();
  instance_norm2d_test();
  // reflection_pad2d_test();
  // unet_test();

#if defined(_MSC_VER)
  std::cin.get();
#endif

  return 0;
}