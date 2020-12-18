#include "module/conv2d.h"
#include "module/prelu.h"
#include "module/tanh.h"
#include "module/sequential.h"
#include "module/upsample2d.h"
#include "module/instance_norm2d.h"
#include "module/reflection_pad2d.h"
#include "math/instance_norm2d.h"
#include "network/face_mesh.h"

namespace dlfm::nn::face_mesh {

BlazeBlock::BlazeBlock(int64_t in_channels, int64_t out_channels, int64_t kernel_size, int64_t s) {
  stride = s;
  channel_pad = out_channels - in_channels;

  int64_t padding = 0;
  if (2 == stride) {
    max_pool = max_pooling2d((size_t)stride, (size_t)stride, 0);
    padding = 0;
  } else {
    padding = (kernel_size - 1) / 2;
  }

  ADD_SUB_MODULE(convs, sequential, {
    conv2d(in_channels, in_channels, kernel_size, stride, padding, 1, in_channels, true),
    conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, true)}
  );

  ADD_SUB_MODULE(act, prelu, true, out_channels);
}

Tensor BlazeBlock::forward(Tensor x) {
  Tensor h = x;

  if (2 == stride) {
    h = x.pad({0, 2, 0, 2});
    x = (*max_pool)(x);
  }

  if (channel_pad > 0) {
    x = x.pad({0, 0, 0, 0, 0, (size_t)channel_pad});
  }

  return (*act)((*convs)(h) + x);
}

FaceMesh::FaceMesh() {
  ADD_SUB_MODULE(backbone, sequential, {
    conv2d(3, 16, 3, 2, 1, 1, 1, true),
    prelu(true, 16),

    std::make_shared<BlazeBlock>(16, 16),
    std::make_shared<BlazeBlock>(16, 16),
    std::make_shared<BlazeBlock>(16, 32, 3, 2),

    std::make_shared<BlazeBlock>(32, 32),
    std::make_shared<BlazeBlock>(32, 32),
    std::make_shared<BlazeBlock>(32, 64, 3, 2),

    std::make_shared<BlazeBlock>(64, 64),
    std::make_shared<BlazeBlock>(64, 64),
    std::make_shared<BlazeBlock>(64, 128, 3, 2),

    std::make_shared<BlazeBlock>(128, 128),
    std::make_shared<BlazeBlock>(128, 128),
    std::make_shared<BlazeBlock>(128, 128, 3, 2),

    std::make_shared<BlazeBlock>(128, 128),
    std::make_shared<BlazeBlock>(128, 128),
    std::make_shared<BlazeBlock>(128, 128, 3, 2),

    std::make_shared<BlazeBlock>(128, 128),
    std::make_shared<BlazeBlock>(128, 128)
  });

  ADD_SUB_MODULE(conv, sequential, {
    conv2d(128, 32, 1, 1, 0, 1, 1, true),
    prelu(true, 32),
  });

  ADD_SUB_MODULE(block, std::make_shared<BlazeBlock>, 32, 32);

  ADD_SUB_MODULE(pred, conv2d, 32, 1404, 3, 3, 0, 1, 1, true);
}

Tensor FaceMesh::forward(Tensor x) {
  x = (*backbone)(x);

  x = (*conv)(x);

  x = (*block)(x);

  return (*pred)(x);
}

}
















