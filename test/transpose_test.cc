#include "common/tensor.h"
#include "transpose_test.h"

namespace dlfm {
namespace test {

void transpose_test() {
  auto t1 = Tensor::create({4, 5});
  auto t2 = t1.transpose({1, 0});

  std::cout << t1 << "\n";
  std::cout << t2 << "\n";
}

}
}