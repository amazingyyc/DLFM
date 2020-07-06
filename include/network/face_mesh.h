#ifndef NN_FACE_MESH_H
#define NN_FACE_MESH_H

#include "module/module.h"
#include "module/sequential.h"
#include "module/prelu.h"
#include "module/conv2d.h"
#include "module/max_pooling2d.h"
#include "module/instance_norm2d.h"
#include "module/reflection_pad2d.h"
#include "module/zero_pad2d.h"
#include "module/linear.h"

namespace dlfm::nn::face_mesh {

class BlazeBlock: public ModuleImpl {
public:
  int64_t stride;
  int64_t channel_pad;

  MaxPooling2d max_pool;

  Sequential convs;

  PRelu act;

  BlazeBlock(int64_t in_channels, int64_t out_channels, int64_t kernel_size=3, int64_t stride=1);

public:
  Tensor forward(Tensor) override;
};

class FaceMesh: public ModuleImpl {
public:
  Sequential backbone;

  Sequential conv;

  std::shared_ptr<BlazeBlock> block;

  Conv2d pred;

  FaceMesh();

public:
  Tensor forward(Tensor) override;
};

}

#endif


















