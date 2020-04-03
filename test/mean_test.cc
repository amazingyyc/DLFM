#include "common/tensor.h"
#include "mean_test.h"

namespace dlfm {
namespace test {

void mean_test() {
  float tp[] = {1, 2, 3, 4};

  auto t1 = Tensor::create_from(tp, {2, 2}, ElementType::from<float>());

  auto t2 = t1.mean({-1, -2}, true);

  std::cout << t1 << "\n";
  std::cout << t2 << "\n";
}

void var_test() {
  float tp[] = { 1, 2, 3, 4 };

  auto t1 = Tensor::create_from(tp, {1, 1, 2, 2 }, ElementType::from<float>());

  auto v = t1.var({-1, -2}, true);

  std::cout << v << "\n";
}

void instance_norm2d_test() {
  float tp[] = { 1, 2, 3, 4, 5, 6, 7, 8};

  auto t1 = Tensor::create_from(tp, { 1, 2, 2, 2 }, ElementType::from<float>());

  auto v = t1.instance_norm2d();

  std::cout << v << "\n";
}

void reflection_pad2d_test() {
  float tp[] = {0, 1, 2, 3, 4, 5, 6, 7, 8 };

  auto t1 = Tensor::create_from(tp, { 1, 1, 3, 3 }, ElementType::from<float>());

  auto v = t1.reflection_pad2d(2);

  std::cout << v << "\n";
}

}
}