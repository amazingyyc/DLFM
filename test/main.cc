#include <iostream>

#include "test.h"

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
  // instance_norm2d_test();
  // reflection_pad2d_test();
  // unet_test();
  // conv2d_test();
  // cartoon_transformer_test();
  // std_test();
  style_transformer_test();

#if defined(_MSC_VER)
  std::cin.get();
#endif

  return 0;
}