#include "common/tensor.h"
#include "test.h"
#include "module/tiny_unet.h"
#include "module/cartoon_transformer.h"

namespace dlfm {
namespace test {

void unet_test() {
  // /Users/yanyuanchi/code/Colorful_AI/colorize
  nn::TinyUNet unet(1, 2);
  unet.torch_name_scope("unet");
  unet.load_torch_model("/Users/yanyuanchi/code/Colorful_AI/colorize/colorize3x3");

  auto ones = Tensor::ones({1, 1, 16, 16});
  auto output = unet(ones);

  std::cout << output << "\n";
}

void cartoon_transformer_test() {
  // /Users/yanyuanchi/code/Colorful_AI/colorize/cartoongan
  nn::CartoonTransformer transformer;
  transformer.torch_name_scope("cartoongan");
  transformer.load_torch_model("/Users/yanyuanchi/code/Colorful_AI/colorize/cartoongan");

  auto ones = Tensor::ones({1, 3, 8, 8});
  auto output = transformer.test(ones);

  std::cout << output << "\n";
}

}
}