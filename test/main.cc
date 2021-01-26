#include <iostream>

#include "test.h"
#include "mean_test.h"
#include "conv2d_test.h"
#include "selfie2anime_test.h"

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
  // style_transformer_test();
  // var_test();
  // selfie2anime_test();
  //cartoon_face_test();
  // human_seg_test();
  // anime_face_test();
  // srgan_test();
  // blaze_face_test();
  // anime_face_tiny_test();
  // deserialize_test();
  // hair_seg_test();
  // segnet_test();
  // pfld_test();
  // slim_test();
  // pfld_lite_test();
  // topk_test();
  h3r_test();

#if defined(_MSC_VER)
  std::cin.get();
#endif

  return 0;
}