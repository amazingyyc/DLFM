#include "common/tensor.h"
#include "pad_test.h"

namespace dlfm {
namespace test {

void pad_test() {
  auto t1 = Tensor::create({1, 1, 1, 1});
  t1.fill(2);

  auto t2 = t1.pad({1, 2, 3, 4});

  std::cout << t1 << "\n";
  std::cout << t2 << "\n";
}

}
}