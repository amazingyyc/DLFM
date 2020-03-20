#include "max_pooling2d_test.h"

#include "common/tensor.h"

namespace dlfm {
namespace test {

void max_pooling2d_test() {
  auto t1 = Tensor::ones({1, 2, 7, 3});
  auto t2 = Tensor::zeros({1, 2, 7, 2});

  auto t3 = t1.cat(t2, 3);

  auto t4 = t3.max_pooling2d({2, 2}, {2, 2}, {1, 1});

  std::cout << t4 << "\n";
}

}
}