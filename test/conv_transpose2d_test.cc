#include "common/tensor.h"
#include "conv_transpose2d_test.h"

namespace dlfm {
namespace test {

void conv_transpose2d_test() {
  auto i1 = Tensor::ones({ 1, 1, 2, 2});
  auto i2 = Tensor::zeros({ 1, 1, 3, 2});

  // auto input = Tensor::ones({ 1, 1, 5, 2 });
  float ip[] = {1, 2, 3, 4, 5, 6};

  auto input = Tensor::create_from(ip, {1, 1, 2, 3}, ElementType::from<float>());

  auto w1 = Tensor::zeros({ 1, 1, 5, 2 });
  auto w2 = Tensor::ones({ 1, 1, 5, 2 });

  // auto weight = Tensor::ones({ 1, 1, 5, 4 });
  float wp[] = {9, 8, 7, 6 ,5 ,4 ,3,2,1, 0, -1, -2};
  auto weight = Tensor::create_from(wp, { 1, 1, 4, 3 }, ElementType::from<float>());

  std::cout << input << "\n";
  std::cout << weight << "\n";

  auto bias = Tensor::zeros({1});

  auto output = input.conv_transpose2d(weight, bias, {2, 2}, {1, 1}, {1, 1});

  std::cout << output << "\n";
}

}
}