#include "common/tensor.h"
#include "cast_test.h"

namespace dlfm {
namespace test {

void cast_test() {
  auto t1 = Tensor::ones({4, 6}, ElementType::from<uint8_t>());
  auto t2 = t1.cast(ElementType::from<float>());
  auto t3 = t2 / 255.0;

  std::cout << t1 << "\n";
  std::cout << t2 << "\n";
  std::cout << t3 << "\n";
}

}
}