#ifndef NN_CONV2D_H
#define NN_CONV2D_H

#include "common/tensor.h"
#include "module/named_parameters.h"
#include "module/module.h"

namespace dlfm::nn {

class Conv2dImpl: public ModuleImpl {
public:
  Tensor weight;
  Tensor bias;

  std::vector<size_t> kernel_size;
  std::vector<size_t> stride;
  std::vector<size_t> padding;
  std::vector<size_t> dilation;

  size_t groups;

  bool has_bias;

public:
  Conv2dImpl(
    int64_t in_channel,
    int64_t out_channel,
    const std::vector<size_t> &kernel_size,
    const std::vector<size_t> &stride,
    const std::vector<size_t> &padding,
    const std::vector<size_t> &dilation,
    size_t groups,
    bool has_bias);

public:
  virtual void load_torch_model(
    const std::unordered_map<std::string, Tensor> &tensor_map,
    std::string parent_name_scop) override;

  Tensor forward(Tensor) override;

};

using Conv2d = std::shared_ptr<Conv2dImpl>;

Conv2d conv2d(
  int64_t in_channel,
  int64_t out_channel,
  const std::vector<size_t> &kernel_size,
  const std::vector<size_t> &stride,
  const std::vector<size_t> &padding,
  const std::vector<size_t> &dilation,
  size_t groups,
  bool has_bias);

Conv2d conv2d(
  int64_t in_channel,
  int64_t out_channel,
  size_t kernel_size,
  size_t stride,
  size_t padding,
  size_t groups,
  size_t dilation,
  bool has_bias);

template <typename... Type>
Conv2d conv2d(int64_t in_channel, int64_t out_channel, size_t kernel_size, Type... args) {
  return conv2d(in_channel, out_channel, { kernel_size , kernel_size }, args...);
}

template <typename... Type>
Conv2d conv2d(
  int64_t in_channel,
  int64_t out_channel,
  const std::vector<size_t> &kernel_size,
  Type... args) {
  std::unordered_map<std::string, ParameterType> args_map;

  // get all args
  fetch_variadic_args(args_map, args...);

  // Now we has fetch all parameters in args_map, will replace default value.
  std::vector<size_t> stride = {1, 1};
  std::vector<size_t> padding = {0, 0};
  std::vector<size_t> dilation = {1, 1};
  size_t groups = 1;
  bool bias = true;

  auto fetch_array_arg = [](const std::unordered_map<std::string, ParameterType> &args_map,
                            const std::string &key,
                            std::vector<size_t> &target) {
    auto it = args_map.find(key);

    if (it != args_map.end()) {
      if (it->second.has_array_val()) {
        ARGUMENT_CHECK(2 == it->second.array_val().size(), key << " size error");

        target = it->second.array_val();
      } else if (it->second.has_number_val()) {
        target = { it->second.number_val(), it->second.number_val() };
      }
    }
  };

  fetch_array_arg(args_map, "stride", stride);
  fetch_array_arg(args_map, "padding", padding);
  fetch_array_arg(args_map, "dilation", dilation);

  auto group_it = args_map.find("groups");
  if (group_it != args_map.end()) {
    if (group_it->second.has_number_val()) {
      groups = group_it->second.number_val();
    }
  }

  auto bias_it = args_map.find("bias");
  if (bias_it != args_map.end()) {
    if (bias_it->second.has_flag_val()) {
      bias = bias_it->second.flag_val();
    }
  }

  return conv2d(
    in_channel,
    out_channel,
    kernel_size,
    stride,
    padding,
    dilation,
    groups,
    bias);
}

}

#endif