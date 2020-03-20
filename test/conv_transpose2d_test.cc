#include "common/tensor.h"
#include "conv_transpose2d_test.h"

namespace dlfm {
namespace test {

void conv_transpose2d_test() {
  auto i1 = Tensor::ones({ 1, 1, 2, 2});
  auto i2 = Tensor::zeros({ 1, 1, 3, 2});

  auto input = i1.cat(i2, 2);

  auto w2 = Tensor::ones({ 1, 2, 5, 2 });
  auto w1 = Tensor::zeros({ 1, 2, 5, 2 });

  auto weight = w1.cat(w2, 3);

  std::cout << weight << "\n";

  auto bias = Tensor::ones({2});

  auto output = input.conv_transpose2d(weight, bias, {2, 2}, {2, 2}, {1, 1});

  std::cout << output << "\n";
}

}
}