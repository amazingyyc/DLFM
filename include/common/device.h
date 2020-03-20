#ifndef DEVICE_H
#define DEVICE_H

#include "basic.h"

#ifdef HAS_NNPACK
#include "nnpack.h"
#include "pthreadpool.h"
#endif

namespace dlfm {

#define MEMORY_ALIGNMENT 64

// every tensor will bind to a device
// for now only support CPU
enum class DeviceType : int16_t {
  Default = 0,
  CPU = Default,
};

class IAllocator {
public:
  // malloc memory from device
  virtual void* allocate(size_t) = 0;

  // free the memory
  virtual void deallocate(void *) = 0;

  // zero the memory
  virtual void zero(void *, size_t) = 0;

  // memory copy
  virtual void memcpy(void *, const void *, size_t) = 0;
};

class CPUAllocator : public IAllocator {
public:
  void* allocate(size_t) override;

  void deallocate(void *) override;

  void zero(void *, size_t) override;

  void memcpy(void *, const void *, size_t) override;
};

class Device {
private:
  DeviceType type_;
  int32_t id_;

  // memory allocator
  std::shared_ptr<IAllocator> allocator_;

  // the Eigen device for using Eigen Tensor
  std::shared_ptr<Eigen::NonBlockingThreadPool> eigen_thread_pool_;
  std::shared_ptr<Eigen::ThreadPoolDevice> eigen_device_;

#ifdef HAS_NNPACK
  // thread for NNPACK
  pthreadpool_t nnpack_threadpool_;
#endif

public:
  explicit Device();

  explicit Device(DeviceType type, int32_t id);

  ~Device();

  bool operator==(const Device &) const;

  DeviceType type();

  int32_t id();

  // get eigen device
  std::shared_ptr<Eigen::ThreadPoolDevice> eigen_device();

#ifdef HAS_NNPACK
  pthreadpool_t nnpack_threadpool();
#endif

  // malloc memory from device
  void* allocate(size_t);

  // free the memory
  void deallocate(void *);

  // zero the memory
  void zero(void *, size_t);

  // memory copy
  void memcpy(void *, const void *, size_t);

  // get the system physical thread count
  int64_t get_sys_thread_count();

public:
  static std::shared_ptr<Device> default_device();
};

}

#endif