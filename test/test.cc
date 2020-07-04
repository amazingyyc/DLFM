#include <network/srgan.h>
#include "common/tensor.h"
#include "test.h"
#include "network/tiny_unet.h"
#include "network/style_transformer.h"
#include "network/cartoon_face.h"
#include "network/human_seg.h"
#include "network/anime_face.h"
#include "network/blaze_face.h"
#include "vision/resample.h"
#include "vision/pad.h"

namespace dlfm {
namespace test {

void unet_test() {
  // /Users/yanyuanchi/code/Colorful_AI/colorize
  nn::tiny_unet::TinyUNet unet(1, 2);
  unet.torch_name_scope("unet");
  unet.load_torch_model("/Users/yanyuanchi/code/Colorful_AI/colorize/colorize3x3");

  auto ones = Tensor::ones({1, 1, 16, 16});
  auto output = unet(ones);

  std::cout << output << "\n";
}

void style_transformer_test() {
  nn::style_transformer::Transformer style_transformer;
  style_transformer.torch_name_scope("transformer");
  style_transformer.load_torch_model("/Users/yanyuanchi/code/StyleTransformer/dlfm_models/1/model");

  auto input = Tensor::create({512, 512, 3}, ElementType::from<uint8_t>());
  input.fill(125);

  auto output = style_transformer(input);

  std::cout << output << "\n";
}

void cartoon_face_test() {
  nn::cartoon_face::CartoonFace cartoon(32, 256, true);
  cartoon.torch_name_scope("cartoon_face");
  cartoon.load_torch_model("/Users/yanyuanchi/code/photo2cartoon/dlfm");

  auto i1 = Tensor::ones({1, 2, 32, 32});
  auto i2 = Tensor::zeros({1, 1, 32, 32});


  auto input = i1.cat(i2, 1);
  input = Tensor::ones({1, 3, 32, 32});

  auto output = cartoon(input);

  std::cout << output << "\n";
}

void human_seg_test() {
  nn::human_seg::HumanSeg seg;
  seg.torch_name_scope("human_seg");
  seg.load_torch_model("/Users/yanyuanchi/code/Human-Segmentation-PyTorch/pretrain_model/dlfm");

  // auto input = Tensor::ones({1, 3, 256, 256});
  auto input = Tensor::create({256, 256, 3}, ElementType::from<uint8_t>());

  auto output = seg(input);

  // [1, 2, 256, 256]
  // std::cout << output.slice({0, 1, 0, 0}, {1, 1, 1, 256}) << "\n";
  auto tt = output[0];
  tt = tt[1];
  tt = tt[0];
  std::cout << tt << "\n";
}

void anime_face_test() {
  nn::anime_face::AnimeFace anime_face(3, 3, 64, 4, 256);
  anime_face.torch_name_scope("anime_face");
  anime_face.load_torch_model("/Users/yanyuanchi/code/UGATIT-pytorch/dlfm");

  auto input = Tensor::ones({1, 3, 256, 256});

  // [1, 3, 256, 256]
  auto output = anime_face(input);

  std::cout << output[0][2][1];
}

void srgan_test() {
  nn::srgan::SRResNet srres_net;
  srres_net.torch_name_scope("srgan.net");
  srres_net.load_torch_model("/Users/yanyuanchi/code/a-PyTorch-Tutorial-to-Super-Resolution/dlfm");

  auto input = Tensor::ones({3, 2, 2});

  auto output = srres_net(input);


  //std::cout << output[0][2][1];
}

void blaze_face_test() {
//  nn::blaze_face::BlazeFace blaze_face;
//  blaze_face.torch_name_scope("blazeface");
//  blaze_face.load_torch_model("/Users/yanyuanchi/code/BlazeFace/dlfm");
//
//  auto input = Tensor::ones({1, 3, 128, 128});
//
//  auto output = blaze_face.detect(input);
//
//  std::cout << output.size() << "\n";
//  auto x = Tensor::create({1280, 1024, 4}, ElementType::from<uint8_t>());
//
//  auto y = vision::resize(x, {128, 128});

  auto white = dlfm::Tensor::create({4}, dlfm::ElementType::from<uint8_t>());
  white.fill(255);

  auto x = Tensor::create({128, 101, 4}, ElementType::from<uint8_t>());
  auto y = vision::pad(x, white, 0, 0, (128 - 101) / 2, (128 - 101) - (128 - 101) / 2);
}

}
}













