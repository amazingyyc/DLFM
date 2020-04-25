#include "common/tensor.h"
#include "network/selfie2anime.h"
#include "selfie2anime_test.h"

namespace dlfm {
namespace test {

void selfie2anime_test() {
  nn::selfie2anime::VAEGen vae_gen(3, 64, 2, 4, "relu", "zero");
  vae_gen.torch_name_scope("selfie2anime");
  vae_gen.load_torch_model("/Users/yanyuanchi/code/StyleTransformer/selfie2anime/dlfm");

  auto input = Tensor::ones({1, 3, 8, 8});

  auto output = vae_gen(input);

  std::cout << output << "\n";
}

}
}