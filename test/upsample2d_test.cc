#include "common/tensor.h"
#include "upsample2d_test.h"

namespace dlfm {
namespace test {

void upsample2d_test() {
  float x[4] = {1, 2, 3, 4};

  auto t1 = Tensor::create_from(x, {1, 1, 2, 2}, ElementType::from<float>());

  auto t3 = t1.upsample2d(2, "nearest");
  auto t4 = t1.upsample2d(2, "bilinear", false);
  auto t5 = t1.upsample2d(2, "bilinear", true);

  std::cout << t3 << "\n";
  std::cout << t4 << "\n";
  std::cout << t5 << "\n";
}

}
}