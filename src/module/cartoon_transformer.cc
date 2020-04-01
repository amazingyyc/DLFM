#include "common/tensor.h"
#include "module/cartoon_transformer.h"
#include "module/max_pooling2d.h"
#include "module/conv2d.h"
#include "module/relu.h"
#include "module/sequential.h"
#include "module/sigmoid.h"

namespace dlfm::nn {

CartoonTransformer::CartoonTransformer() {
  conv01_1 = conv2d(3, 64, 7);
  in01_1 = instance_norm2d(64);

  conv02_1 = conv2d(64, 128, 3, 2, 1);
  conv02_2 = conv2d(128, 128, 3, 1, 1);
  in02_1 = instance_norm2d(128);

  conv03_1 = conv2d(128, 256, 3, 2, 1);
  conv03_2 = conv2d(256, 256, 3, 1, 1);
  in03_1 = instance_norm2d(256);

  conv04_1 = conv2d(256, 256, 3);
  in04_1 = instance_norm2d(256);
  conv04_2 = conv2d(256, 256, 3);
  in04_2 = instance_norm2d(256);

  conv05_1 = conv2d(256, 256, 3);
  in05_1 = instance_norm2d(256);
  conv05_2 = conv2d(256, 256, 3);
  in05_2 = instance_norm2d(256);

  conv06_1 = conv2d(256, 256, 3);
  in06_1 = instance_norm2d(256);
  conv06_2 = conv2d(256, 256, 3);
  in06_2 = instance_norm2d(256);

  conv07_1 = conv2d(256, 256, 3);
  in07_1 = instance_norm2d(256);
  conv07_2 = conv2d(256, 256, 3);
  in07_2 = instance_norm2d(256);

  conv08_1 = conv2d(256, 256, 3);
  in08_1 = instance_norm2d(256);
  conv08_2 = conv2d(256, 256, 3);
  in08_2 = instance_norm2d(256);

  conv09_1 = conv2d(256, 256, 3);
  in09_1 = instance_norm2d(256);
  conv09_2 = conv2d(256, 256, 3);
  in09_2 = instance_norm2d(256);

  conv10_1 = conv2d(256, 256, 3);
  in10_1 = instance_norm2d(256);
  conv10_2 = conv2d(256, 256, 3);
  in10_2 = instance_norm2d(256);

  conv11_1 = conv2d(256, 256, 3);
  in11_1 = instance_norm2d(256);
  conv11_2 = conv2d(256, 256, 3);
  in11_2 = instance_norm2d(256);

  deconv01_1 = conv_tranpose2d(256, 128, 3, 2, 1, 1);
  deconv01_2 = conv2d(128, 128, 3, 1, 1);
  in12_1 = instance_norm2d(128);

  deconv02_1 = conv_tranpose2d(128, 64, 3, 2, 1, 1);
  deconv02_2 = conv2d(64, 64, 3, 1, 1);
  in13_1 = instance_norm2d(64);

  deconv03_1 = conv2d(64, 3, 7);
}

void CartoonTransformer::torch_name_scope(std::string name) {
  torch_name_scope_ = name;

  conv01_1->torch_name_scope(name + TORCH_NAME_SCOPE_SEP + "conv01_1");
  in01_1->torch_name_scope(name + TORCH_NAME_SCOPE_SEP + "in01_1");

  conv02_1->torch_name_scope(name + TORCH_NAME_SCOPE_SEP + "conv02_1");
  conv02_2->torch_name_scope(name + TORCH_NAME_SCOPE_SEP + "conv02_2");
  in02_1->torch_name_scope(name + TORCH_NAME_SCOPE_SEP + "in02_1");

  conv03_1->torch_name_scope(name + TORCH_NAME_SCOPE_SEP + "conv03_1");
  conv03_2->torch_name_scope(name + TORCH_NAME_SCOPE_SEP + "conv03_2");
  in03_1->torch_name_scope(name + TORCH_NAME_SCOPE_SEP + "in03_1");

  conv04_1->torch_name_scope(name + TORCH_NAME_SCOPE_SEP + "conv04_1");
  in04_1->torch_name_scope(name + TORCH_NAME_SCOPE_SEP + "in04_1");
  conv04_2->torch_name_scope(name + TORCH_NAME_SCOPE_SEP + "conv04_2");
  in04_2->torch_name_scope(name + TORCH_NAME_SCOPE_SEP + "in04_2");

  conv05_1->torch_name_scope(name + TORCH_NAME_SCOPE_SEP + "conv05_1");
  in05_1->torch_name_scope(name + TORCH_NAME_SCOPE_SEP + "in05_1");
  conv05_2->torch_name_scope(name + TORCH_NAME_SCOPE_SEP + "conv05_2");
  in05_2->torch_name_scope(name + TORCH_NAME_SCOPE_SEP + "in05_2");

  conv06_1->torch_name_scope(name + TORCH_NAME_SCOPE_SEP + "conv06_1");
  in06_1->torch_name_scope(name + TORCH_NAME_SCOPE_SEP + "in06_1");
  conv06_2->torch_name_scope(name + TORCH_NAME_SCOPE_SEP + "conv06_2");
  in06_2->torch_name_scope(name + TORCH_NAME_SCOPE_SEP + "in06_2");

  conv07_1->torch_name_scope(name + TORCH_NAME_SCOPE_SEP + "conv07_1");
  in07_1->torch_name_scope(name + TORCH_NAME_SCOPE_SEP + "in07_1");
  conv07_2->torch_name_scope(name + TORCH_NAME_SCOPE_SEP + "conv07_2");
  in07_2->torch_name_scope(name + TORCH_NAME_SCOPE_SEP + "in07_2");

  conv08_1->torch_name_scope(name + TORCH_NAME_SCOPE_SEP + "conv08_1");
  in08_1->torch_name_scope(name + TORCH_NAME_SCOPE_SEP + "in08_1");
  conv08_2->torch_name_scope(name + TORCH_NAME_SCOPE_SEP + "conv08_2");
  in08_2->torch_name_scope(name + TORCH_NAME_SCOPE_SEP + "in08_2");

  conv09_1->torch_name_scope(name + TORCH_NAME_SCOPE_SEP + "conv09_1");
  in09_1->torch_name_scope(name + TORCH_NAME_SCOPE_SEP + "in09_1");
  conv09_2->torch_name_scope(name + TORCH_NAME_SCOPE_SEP + "conv09_2");
  in09_2->torch_name_scope(name + TORCH_NAME_SCOPE_SEP + "in09_2");

  conv10_1->torch_name_scope(name + TORCH_NAME_SCOPE_SEP + "conv10_1");
  in10_1->torch_name_scope(name + TORCH_NAME_SCOPE_SEP + "in10_1");
  conv10_2->torch_name_scope(name + TORCH_NAME_SCOPE_SEP + "conv10_2");
  in10_2->torch_name_scope(name + TORCH_NAME_SCOPE_SEP + "in10_2");

  conv11_1->torch_name_scope(name + TORCH_NAME_SCOPE_SEP + "conv11_1");
  in11_1->torch_name_scope(name + TORCH_NAME_SCOPE_SEP + "in11_1");
  conv11_2->torch_name_scope(name + TORCH_NAME_SCOPE_SEP + "conv11_2");
  in11_2->torch_name_scope(name + TORCH_NAME_SCOPE_SEP + "in11_2");

  deconv01_1->torch_name_scope(name + TORCH_NAME_SCOPE_SEP + "deconv01_1");
  deconv01_2->torch_name_scope(name + TORCH_NAME_SCOPE_SEP + "deconv01_2");
  in12_1->torch_name_scope(name + TORCH_NAME_SCOPE_SEP + "in12_1");

  deconv02_1->torch_name_scope(name + TORCH_NAME_SCOPE_SEP + "deconv02_1");
  deconv02_2->torch_name_scope(name + TORCH_NAME_SCOPE_SEP + "deconv02_2");
  in13_1->torch_name_scope(name + TORCH_NAME_SCOPE_SEP + "in13_1");

  deconv03_1->torch_name_scope(name + TORCH_NAME_SCOPE_SEP + "deconv03_1");
}

std::vector<Module> CartoonTransformer::sub_modules() {
  return {
      conv01_1;
      in01_1;

      conv02_1;
      conv02_2;
      in02_1;

      conv03_1;
      conv03_2;
      in03_1;

      conv04_1;
      in04_1;
      conv04_2;
      in04_2;

      conv05_1;
      in05_1;
      conv05_2;
      in05_2;

      conv06_1;
      in06_1;
      conv06_2;
      in06_2;

      conv07_1;
      in07_1;
      conv07_2;
      in07_2;

      conv08_1;
      in08_1;
      conv08_2;
      in08_2;

      conv09_1;
      in09_1;
      conv09_2;
      in09_2;

      conv10_1;
      in10_1;
      conv10_2;
      in10_2;

      conv11_1;
      in11_1;
      conv11_2;
      in11_2;

      deconv01_1;
      deconv01_2;
      in12_1;

      deconv02_1;
      deconv02_2;
      in13_1;

      deconv03_1;
  };
}

Tensor CartoonTransformer::forward(Tensor x) {

}

}