#ifndef DESERIALIZE_H_KK
#define DESERIALIZE_H_KK

#include "basic.h"
#include "shape.h"
#include "tensor_storage.h"
#include "element_type.h"
#include "tensor.h"

// A very simple deserialize api use to deserialize model file.
namespace dlfm {

class StreamReader {
public:
  virtual bool read(void *byte, size_t length) = 0;
};

class FileStreamReader: public StreamReader {
private:
  std::ifstream fstream_;

  std::string file_path_;

public:
  FileStreamReader(const std::string &file_path);

  ~FileStreamReader();

  bool read(void *byte, size_t length) override;
};

class ModelDeserialize {
private:
  FileStreamReader reader_;

public:
  ModelDeserialize(const std::string &file_path);

  template<typename T>
  bool deserialize(T &val) {
    return false;
  }
};

// Basic Type.
#undef DEFINE_BASIC_TYPE_DESERIALIZE
#define DEFINE_BASIC_TYPE_DESERIALIZE(Type)                   \
template<>                                                    \
bool inline ModelDeserialize::deserialize<Type>(Type &val) {  \
  return reader_.read(((void*)(&val)), sizeof(Type));         \
}                                                             \

DEFINE_BASIC_TYPE_DESERIALIZE(bool);
DEFINE_BASIC_TYPE_DESERIALIZE(uint8_t);
DEFINE_BASIC_TYPE_DESERIALIZE(int8_t);
DEFINE_BASIC_TYPE_DESERIALIZE(uint16_t);
DEFINE_BASIC_TYPE_DESERIALIZE(int16_t);
DEFINE_BASIC_TYPE_DESERIALIZE(uint32_t);
DEFINE_BASIC_TYPE_DESERIALIZE(int32_t);
DEFINE_BASIC_TYPE_DESERIALIZE(uint64_t);
DEFINE_BASIC_TYPE_DESERIALIZE(int64_t);
DEFINE_BASIC_TYPE_DESERIALIZE(float);
DEFINE_BASIC_TYPE_DESERIALIZE(double);

#undef DEFINE_BASIC_TYPE_DESERIALIZE

template<>
bool inline ModelDeserialize::deserialize<std::string>(std::string &val) {
  uint64_t size;

  if (!deserialize<uint64_t>(size)) {
    return false;
  }

  val.resize(size);

  return reader_.read((void*)(val.data()), size);
}

template<>
bool inline ModelDeserialize::deserialize<Shape>(Shape &shape) {
  uint64_t rank;

  if (!deserialize<uint64_t>(rank)) {
    return false;
  }

  std::vector<int64_t> dims(rank);

  for (int64_t i = 0; i < rank; ++i) {
    if (!deserialize<int64_t>(dims[i])) {
      return false;
    }
  }

  shape.reset(dims);

  return true;
}

template<>
bool inline ModelDeserialize::deserialize<ElementType>(ElementType &element_type) {
  int16_t dtype;

  if (!deserialize<int16_t>(dtype)) {
    return false;
  }

  switch (dtype) {
    case 0:
      element_type = ElementType::from<UnKnownType>();
      return true;
    case 1:
      element_type = ElementType::from<bool>();
      return true;
    case 2:
      element_type = ElementType::from<uint8_t>();
      return true;
    case 3:
      element_type = ElementType::from<int8_t>();
      return true;
    case 4:
      element_type = ElementType::from<uint16_t>();
      return true;
    case 5:
      element_type = ElementType::from<int16_t>();
      return true;
    case 6:
      element_type = ElementType::from<uint32_t>();
      return true;
    case 7:
      element_type = ElementType::from<int32_t>();
      return true;
    case 8:
      element_type = ElementType::from<uint64_t>();
      return true;
    case 9:
      element_type = ElementType::from<int64_t>();
      return true;
    case 10:
      element_type = ElementType::from<float>();
      return true;
    case 11:
      element_type = ElementType::from<double>();
      return true;
    case 12:
      element_type = ElementType::from<float16>();
      return true;
    default:
      return false;
  }
}

template<>
bool inline ModelDeserialize::deserialize<Tensor>(Tensor &tensor) {
  // read step: element_type, shape, offset, storage
  ElementType element_type = ElementType::from<UnKnownType>();
  Shape shape;
  uint64_t offset;

  if (!deserialize<ElementType>(element_type) ||
      !deserialize<Shape>(shape) ||
      !deserialize<uint64_t>(offset)) {
    return false;
  }

  auto storage = TensorStorage::create(shape.size() * element_type.byte_width());

  if (!reader_.read(storage->ptr(), storage->size())) {
    return false;
  }

  tensor = Tensor::create(storage, offset, shape, element_type);

  return true;
}

template<>
bool inline ModelDeserialize::deserialize<std::unordered_map<std::string, Tensor>>(std::unordered_map<std::string, Tensor> &tensor_map) {
  tensor_map.clear();

  uint64_t pair_count;

  if (!deserialize<uint64_t>(pair_count)) {
    return false;
  }

  for (uint64_t i = 0; i < pair_count; ++i) {
    std::string name;
    Tensor tensor;

    if (!deserialize<std::string>(name) || !deserialize<Tensor>(tensor)) {
      return false;
    }

    tensor_map.insert(std::make_pair(name, tensor));
  }

  return true;
}

}

#endif