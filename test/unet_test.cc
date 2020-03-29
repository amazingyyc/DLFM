#include <module/tiny_unet.h>
#include "unet_test.h"

namespace dlfm {
namespace test {

void unet_test() {
  std::string folder = "/Users/yanyuanchi/code/Colorful_AI/colorize/colorize3x3";

  auto unet = nn::TinyUNet(1, 2);
  unet.torch_name_scope("unet");
  unet.load_torch_model(folder);

  auto input = Tensor::ones({1, 1, 17, 16});
  auto output = unet(input);

  std::cout << output << "\n";
}

}
}
