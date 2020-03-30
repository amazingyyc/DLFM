#include "common/tensor.h"

namespace dlfm {
namespace test {

void matmul_test() {
  auto x = Tensor::ones({ 3, 2 });
  auto y = Tensor::ones({ 2, 3 });

  auto z = x.matmul(y, true, true);

  std::cout << z << "\n";
}

}
}