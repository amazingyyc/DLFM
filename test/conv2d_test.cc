#include "common/tensor.h"
#include "conv2d_test.h"

namespace dlfm {
namespace test {

void conv2d_test() {
  float ip[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  auto input  = Tensor::create_from(ip, {1, 1, 4, 4}, ElementType::from<float>());
  auto weight = Tensor::ones({ 2, 1, 3, 3 });
  auto bias = Tensor::zeros({2});

  auto output = input.conv2d(weight, bias, {1, 1}, {0, 0});

  std::cout << output << "\n";
}

}
}