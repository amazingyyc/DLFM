#include "common/exception.h"
#include "common/tensor_storage.h"

namespace dlfm {

TensorStorage::TensorStorage(std::shared_ptr<Device> device, void *ptr, size_t size, bool own)
  : device_(device), ptr_(ptr), size_(size), own_(own) {
  }

TensorStorage::~TensorStorage() {
  if (own_.load()) {
    device_->deallocate(ptr_);
  }

  ptr_ = nullptr;
  size_ = 0;
}

std::shared_ptr<Device> TensorStorage::device() {
  return device_;
}

void* TensorStorage::ptr() {
  return ptr_;
}

size_t TensorStorage::size() {
  return size_;
}

bool TensorStorage::own() {
  return own_.load();
}

std::shared_ptr<TensorStorage> TensorStorage::create(size_t size) {
  ARGUMENT_CHECK(size > 0, "can not malloc < 0 bytes memory");

  auto device = Device::default_device();

  void *ptr = device->allocate(size);

  return std::make_shared<TensorStorage>(device, ptr, size, true);
}

std::shared_ptr<TensorStorage> TensorStorage::create_from(void *ptr, size_t size) {
  ARGUMENT_CHECK(size > 0, "can not create a 0 TensorStorage");

  auto device = Device::default_device();

  return std::make_shared<TensorStorage>(device, ptr, size, false);
}

}