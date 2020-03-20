#include "common/tensor.h"
#include "cat_test.h"

namespace dlfm {
namespace test {

void cat_test() {
  auto t1 = Tensor::ones({2, 1, 2});
  auto t2 = Tensor::zeros({2, 2, 2});

  auto t3 = t1.cat(t2, 1);

  std::cout << t3 << "\n";
}

}
}