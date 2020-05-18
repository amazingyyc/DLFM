#include "common/tensor.h"
#include "test.h"
#include "network/tiny_unet.h"
#include "network/style_transformer.h"
#include "network/cartoon_face.h"
#include "network/human_seg.h"
#include "network/anime_face.h"

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

}
}