#include "common/tensor.h"
#include "module/cartoon_transformer.h"
#include "module/max_pooling2d.h"
#include "module/conv2d.h"
#include "module/relu.h"
#include "module/sequential.h"
#include "module/sigmoid.h"

namespace dlfm::nn {

CartoonTransformer::CartoonTransformer() {
  ADD_SUB_MODULE(conv01_1, conv2d, 3, 64, 7);
  ADD_SUB_MODULE(in01_1, instance_norm2d, 64);

  ADD_SUB_MODULE(conv02_1, conv2d, 64, 128, 3, 2, 1);
  ADD_SUB_MODULE(conv02_2, conv2d, 128, 128, 3, 1, 1);
  ADD_SUB_MODULE(in02_1, instance_norm2d, 128);

  ADD_SUB_MODULE(conv03_1, conv2d, 128, 256, 3, 2, 1);
  ADD_SUB_MODULE(conv03_2, conv2d, 256, 256, 3, 1, 1);
  ADD_SUB_MODULE(in03_1, instance_norm2d, 256);

  ADD_SUB_MODULE(conv04_1, conv2d, 256, 256, 3);
  ADD_SUB_MODULE(in04_1, instance_norm2d, 256);
  ADD_SUB_MODULE(conv04_2, conv2d, 256, 256, 3);
  ADD_SUB_MODULE(in04_2, instance_norm2d, 256);

  ADD_SUB_MODULE(conv05_1, conv2d, 256, 256, 3);
  ADD_SUB_MODULE(in05_1, instance_norm2d, 256);
  ADD_SUB_MODULE(conv05_2, conv2d, 256, 256, 3);
  ADD_SUB_MODULE(in05_2, instance_norm2d, 256);

  ADD_SUB_MODULE(conv06_1, conv2d, 256, 256, 3);
  ADD_SUB_MODULE(in06_1, instance_norm2d, 256);
  ADD_SUB_MODULE(conv06_2, conv2d, 256, 256, 3);
  ADD_SUB_MODULE(in06_2, instance_norm2d, 256);

  ADD_SUB_MODULE(conv07_1, conv2d, 256, 256, 3);
  ADD_SUB_MODULE(in07_1, instance_norm2d, 256);
  ADD_SUB_MODULE(conv07_2, conv2d, 256, 256, 3);
  ADD_SUB_MODULE(in07_2, instance_norm2d, 256);

  ADD_SUB_MODULE(conv08_1, conv2d, 256, 256, 3);
  ADD_SUB_MODULE(in08_1, instance_norm2d, 256);
  ADD_SUB_MODULE(conv08_2, conv2d, 256, 256, 3);
  ADD_SUB_MODULE(in08_2, instance_norm2d, 256);

  ADD_SUB_MODULE(conv09_1, conv2d, 256, 256, 3);
  ADD_SUB_MODULE(in09_1, instance_norm2d, 256);
  ADD_SUB_MODULE(conv09_2, conv2d, 256, 256, 3);
  ADD_SUB_MODULE(in09_2, instance_norm2d, 256);

  ADD_SUB_MODULE(conv10_1, conv2d, 256, 256, 3);
  ADD_SUB_MODULE(in10_1, instance_norm2d, 256);
  ADD_SUB_MODULE(conv10_2, conv2d, 256, 256, 3);
  ADD_SUB_MODULE(in10_2, instance_norm2d, 256);

  ADD_SUB_MODULE(conv11_1, conv2d, 256, 256, 3);
  ADD_SUB_MODULE(in11_1, instance_norm2d, 256);
  ADD_SUB_MODULE(conv11_2, conv2d, 256, 256, 3);
  ADD_SUB_MODULE(in11_2, instance_norm2d, 256);

  ADD_SUB_MODULE(deconv01_1, conv_tranpose2d, 256, 128, 3, 2, 1, 1);
  ADD_SUB_MODULE(deconv01_2, conv2d, 128, 128, 3, 1, 1);
  ADD_SUB_MODULE(in12_1, instance_norm2d, 128);

  ADD_SUB_MODULE(deconv02_1, conv_tranpose2d, 128, 64, 3, 2, 1, 1);
  ADD_SUB_MODULE(deconv02_2, conv2d, 64, 64, 3, 1, 1);
  ADD_SUB_MODULE(in13_1, instance_norm2d, 64);

  ADD_SUB_MODULE(deconv03_1, conv2d, 64, 3, 7);
}

Tensor CartoonTransformer::test(Tensor x) {
  auto yy =  ((*conv01_1)(x.reflection_pad2d(3)));
  return (*in01_1)(yy);
}

Tensor CartoonTransformer::forward(Tensor x) {
  Tensor y, pre;

  y   = ((*in01_1)((*conv01_1)(x.reflection_pad2d(3)))).relu(true);

  y   = ((*in02_1)((*conv02_2)((*conv02_1)(y)))).relu(true);
  pre = ((*in03_1)((*conv03_2)((*conv03_1)(y)))).relu(true);

  y   = ((*in04_1)((*conv04_1)(pre.reflection_pad2d(1)))).relu(true);
  pre = (*in04_2)((*conv04_2)(y.reflection_pad2d(1))) + pre;

  y   = ((*in05_1)((*conv05_1)(pre.reflection_pad2d(1)))).relu(true);
  pre = (*in05_2)((*conv05_2)(y.reflection_pad2d(1))) + pre;

  return pre;

  y   = ((*in06_1)((*conv06_1)(pre.reflection_pad2d(1)))).relu(true);
  pre = (*in06_2)((*conv06_2)(y.reflection_pad2d(1))) + pre;

  y   = ((*in07_1)((*conv07_1)(pre.reflection_pad2d(1)))).relu(true);
  pre = (*in07_2)((*conv07_2)(y.reflection_pad2d(1))) + pre;

  y   = ((*in08_1)((*conv08_1)(pre.reflection_pad2d(1)))).relu(true);
  pre = (*in08_2)((*conv08_2)(y.reflection_pad2d(1))) + pre;

  y   = ((*in09_1)((*conv09_1)(pre.reflection_pad2d(1)))).relu(true);
  pre = (*in09_2)((*conv09_2)(y.reflection_pad2d(1))) + pre;

  y   = ((*in10_1)((*conv10_1)(pre.reflection_pad2d(1)))).relu(true);
  pre = (*in10_2)((*conv10_2)(y.reflection_pad2d(1))) + pre;

  y = ((*in11_1)((*conv11_1)(pre.reflection_pad2d(1)))).relu(true);
  y = (*in11_2)((*conv11_2)(y.reflection_pad2d(1))) + pre;

  y = ((*in12_1)((*deconv01_2)((*deconv01_1)(y)))).relu(true);
  y = ((*in13_1)((*deconv02_2)((*deconv02_1)(y)))).relu(true);

  y = (*deconv03_1)(y.reflection_pad2d(3)).tanh(true);

  return y;
}

}