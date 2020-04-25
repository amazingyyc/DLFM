#include "common/tensor.h"
#include "test.h"
#include "network/tiny_unet.h"
#include "network/style_transformer.h"

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

  //std::cout << output << "\n";
}

}
}