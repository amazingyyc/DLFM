#include "common/tensor.h"
#include "conv2d_test.h"

namespace dlfm {
namespace test {

void conv2d_test() {
  auto input  = Tensor::ones({ 1, 1, 4, 4});
  auto weight = Tensor::ones({ 2, 1, 3, 3 });
  auto bias = Tensor::ones({2});

  auto output = input.conv2d(weight, bias, {1, 1}, {1, 1});

  std::cout << output << "\n";
}

}
}