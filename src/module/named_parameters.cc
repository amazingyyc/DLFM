#include "module/named_parameters.h"

namespace dlfm::nn {

ParameterType::ParameterType(const std::string &key): key_(key), has_val_(0) {
}

const std::string& ParameterType::key() {
  return key_;
}

bool ParameterType::flag_val() const {
  return (bool)number_val_;
}

bool ParameterType::has_flag_val() const {
  return has_number_val();
}

size_t ParameterType::number_val() const {
  return number_val_;
}

bool ParameterType::has_number_val() const {
  return has_val_ & 0x1;
}

const std::vector<size_t>& ParameterType::array_val() const {
  return array_val_;
}

bool ParameterType::has_array_val() const {
  return has_val_ & 0x2;
}

void ParameterType::number_val(size_t number_val) {
  has_val_ |= 0x1;

  number_val_ = number_val;
}

void ParameterType::array_val(const std::vector<size_t> &array_val) {
  has_val_ |= 0x2;

  array_val_.clear();
  array_val_.insert(array_val_.end(), array_val.begin(), array_val.end());
}

NamedParameterIndicator::NamedParameterIndicator(const std::string &k): key(k) {
}

ParameterType NamedParameterIndicator::operator=(size_t number_val) const {
  ParameterType parameter(key);
  parameter.number_val(number_val);

  return parameter;
}

ParameterType NamedParameterIndicator::operator=(const std::vector<size_t> &array_val) const {
  ParameterType parameter(key);
  parameter.array_val(array_val);

  return parameter;
}

const NamedParameterIndicator kernel_size = NamedParameterIndicator("kernel_size");
const NamedParameterIndicator stride = NamedParameterIndicator("stride");
const NamedParameterIndicator padding = NamedParameterIndicator("padding");
const NamedParameterIndicator dilation = NamedParameterIndicator("dilation");
const NamedParameterIndicator groups = NamedParameterIndicator("groups");
const NamedParameterIndicator bias = NamedParameterIndicator("bias");

}