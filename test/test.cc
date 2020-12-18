#include <network/srgan.h>
#include "common/tensor.h"
#include "test.h"
#include "network/tiny_unet.h"
#include "network/style_transformer.h"
#include "network/cartoon_face.h"
#include "network/human_seg.h"
#include "network/anime_face.h"
#include "network/blaze_face.h"
#include "vision/resample.h"
#include "vision/pad.h"
#include "network/face_mesh.h"
#include "network/anime_face_tiny.h"
#include "network/hair_seg.h"
#include "common/deserialize.h"

namespace dlfm {
namespace test {

void unet_test() {
  // /Users/yanyuanchi/code/Colorful_AI/colorize
  nn::tiny_unet::TinyUNet unet(1, 2);
  unet.torch_name_scope("unet");
  // unet.load_torch_model("/Users/yanyuanchi/code/Colorful_AI/colorize/colorize3x3");

  auto ones = Tensor::ones({1, 1, 16, 16});
  auto output = unet(ones);

  std::cout << output << "\n";
}

void style_transformer_test() {
  nn::style_transformer::Transformer style_transformer;
  style_transformer.torch_name_scope("transformer");
  // style_transformer.load_torch_model("/Users/yanyuanchi/code/StyleTransformer/dlfm_models/1/model");

  auto input = Tensor::create({512, 512, 3}, ElementType::from<uint8_t>());
  input.fill(125);

  auto output = style_transformer(input);

  std::cout << output << "\n";
}

void cartoon_face_test() {
  nn::cartoon_face::CartoonFace cartoon(32, 256, true);
  cartoon.torch_name_scope("cartoon_face");
  // cartoon.load_torch_model("/Users/yanyuanchi/code/photo2cartoon/dlfm");

  auto i1 = Tensor::ones({1, 2, 32, 32});
  auto i2 = Tensor::zeros({1, 1, 32, 32});


  auto input = i1.cat(i2, 1);
  input = Tensor::ones({1, 3, 32, 32});

  auto output = cartoon(input);

  std::cout << output << "\n";
}

void human_seg_test() {
  nn::human_seg::HumanSeg seg;
  seg.torch_name_scope("human_seg");
  // seg.load_torch_model("/Users/yanyuanchi/code/Human-Segmentation-PyTorch/pretrain_model/dlfm");

  // auto input = Tensor::ones({1, 3, 256, 256});
  auto input = Tensor::create({256, 256, 3}, ElementType::from<uint8_t>());

  auto output = seg(input);

  // [1, 2, 256, 256]
  // std::cout << output.slice({0, 1, 0, 0}, {1, 1, 1, 256}) << "\n";
  auto tt = output[0];
  tt = tt[1];
  tt = tt[0];
  std::cout << tt << "\n";
}

void anime_face_test() {
  std::string path("/Users/yanyuanchi/code/SerializePytorchModel/anime_face/anime_face_0.pt");

  ModelDeserialize model_deserialize(path);

  std::unordered_map<std::string, Tensor> tensor_map;

  // read model
  model_deserialize.deserialize(tensor_map);

  nn::anime_face::AnimeFace anime_face(3, 3, 64, 4, 256);
  anime_face.load_torch_model(tensor_map);

  auto input = Tensor::ones({1, 3, 256, 256});

  // [1, 3, 256, 256]
  auto output = anime_face(input);

  std::cout << output[0][2][1];
}

void srgan_test() {
  nn::srgan::SRResNet srres_net;
  srres_net.torch_name_scope("srgan.net");
  // srres_net.load_torch_model("/Users/yanyuanchi/code/a-PyTorch-Tutorial-to-Super-Resolution/dlfm");

  auto input = Tensor::ones({3, 2, 2});

  auto output = srres_net(input);


  //std::cout << output[0][2][1];
}

void blaze_face_test() {
  nn::face_mesh::FaceMesh faceMesh;
  faceMesh.torch_name_scope("facemesh");
  // faceMesh.load_torch_model("/Users/yanyuanchi/code/BlazeFace/facemesh/dlfm");

  auto input = Tensor::create({1, 3, 192, 192});
  input.initialize_from_file("/Users/yanyuanchi/code/BlazeFace/facemesh/test_img.bin");

  auto output = faceMesh(input);

  std::cout << output << "\n";
}

void anime_face_tiny_test() {
  nn::anime_face_tiny::AnimeFaceTiny anime_face_tiny(3, 3);
  anime_face_tiny.torch_name_scope("anime_face_tiny");
  // anime_face_tiny.load_torch_model("/Users/yanyuanchi/code/UGATIT-pytorch/dlfm_tiny/cartoon_face_1");

  auto input = Tensor::ones({1, 3, 256, 256});

  // [1, 3, 256, 256]
  auto output = anime_face_tiny(input);

  std::cout << output[0][2][1];
}

void deserialize_test() {
  std::string path("/Users/yanyuanchi/code/SerializePytorchModel/test.bin");

  ModelDeserialize model_deserialize(path);

  std::unordered_map<std::string, Tensor> tensor_map;

  model_deserialize.deserialize(tensor_map);
}

void hair_seg_test() {
  nn::hair_seg::HairSeg hairseg(3, 1);

  std::string path = "/Users/yanyuanchi/code/SerializePytorchModel/hairseg/dlfm/hairseg.dlfm";
  ModelDeserialize model_deserialize(path);

  std::unordered_map<std::string, Tensor> tensor_map;
  model_deserialize.deserialize(tensor_map);

  hairseg.load_torch_model(tensor_map);

  auto input = Tensor::ones({1, 3, 256, 256});

  // [1, 3, 256, 256]
  auto output = hairseg(input);

   std::cout << output << "\n";
   std::cout << output.sum() << "\n";
}

}
}


