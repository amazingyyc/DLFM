#ifndef TENSOR_STORAGE_H
#define TENSOR_STORAGE_H

#include "basic.h"
#include "device.h"

namespace dlfm {

class TensorStorage : public std::enable_shared_from_this<TensorStorage> {
private:
  // the device that allocate the memory
  std::shared_ptr<Device> device_;

  // memory pointer
  void *ptr_;

  // the memory size
  size_t size_;

  // if true mean the memory is malloc from device
  // false means it come from outside (should maitained by user)
  std::atomic<bool> own_;

public:
  TensorStorage(std::shared_ptr<Device> device, void *, size_t, bool);

  ~TensorStorage();

  std::shared_ptr<Device> device();

  void *ptr();

  size_t size();

  bool own();

public:
  static std::shared_ptr<TensorStorage> create(size_t size);

  static std::shared_ptr<TensorStorage> create_from(void *ptr, size_t size);
};

}

#endif