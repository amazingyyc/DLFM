#include "common/exception.h"
#include "common/device.h"

namespace dlfm {

void* CPUAllocator::allocate(size_t num_bytes) {
#ifdef __GNUC__
  void *memory = nullptr;

  if (0 != posix_memalign(&memory, MEMORY_ALIGNMENT, num_bytes)) {
    RUNTIME_ERROR("allocate memory error");
  }

  return memory;

#elif defined(_MSC_VER)
  return _aligned_malloc_dbg(num_bytes, MEMORY_ALIGNMENT, __FILE__, __LINE__);
#else
  RUNTIME_ERROR("not support compiler");
#endif

}

void CPUAllocator::deallocate(void* ptr) {
#ifdef __GNUC__
  std::free(ptr);
#elif defined(_MSC_VER)
  _aligned_free_dbg(ptr);
#else
  RUNTIME_ERROR("not support compiler");
#endif
}

void CPUAllocator::zero(void* ptr, size_t num_bytes) {
  std::memset(ptr, 0, num_bytes);
}

void CPUAllocator::memcpy(void* dest, const void* src, size_t num_bytes) {
  std::memcpy(dest, src, num_bytes);
}

Device::Device() : Device(DeviceType::CPU, 0) {
}

Device::Device(DeviceType type, int32_t id) : type_(type), id_(id) {
  if (type_ == DeviceType::CPU) {
    allocator_ = std::make_shared<CPUAllocator>();
  } else {
    RUNTIME_ERROR("device type error!");
  }

  auto thread_count = get_sys_thread_count();

  eigen_thread_pool_ = std::make_shared<Eigen::NonBlockingThreadPool>(thread_count);
  eigen_device_ = std::make_shared<Eigen::ThreadPoolDevice>(eigen_thread_pool_.get(), thread_count);

#ifdef HAS_NNPACK
  nnpack_threadpool_ = pthreadpool_create(thread_count);
  nnp_initialize();
#endif

  // eigen multi-thread
  Eigen::initParallel();
}

Device::~Device() {
#ifdef HAS_NNPACK
  pthreadpool_destroy(nnpack_threadpool_);
  nnp_deinitialize();
#endif
}

bool Device::operator==(const Device& other) const {
  return type_ == other.type_ && id_ == other.id_;
}

DeviceType Device::type() {
  return type_;
}

int32_t Device::id() {
  return id_;
}

// get eigen device
std::shared_ptr<Eigen::ThreadPoolDevice> Device::eigen_device() {
  return eigen_device_;
}

#ifdef HAS_NNPACK
pthreadpool_t Device::nnpack_threadpool() {
  return nnpack_threadpool_;
}
#endif

// malloc memory from device
void* Device::allocate(size_t num_bytes) {
  return allocator_->allocate(num_bytes);
}

// free the memory
void Device::deallocate(void* ptr) {
  allocator_->deallocate(ptr);
}

// zero the memory
void Device::zero(void* ptr, size_t num_bytes) {
  allocator_->zero(ptr, num_bytes);
}

// memory copy
void Device::memcpy(void* dest, const void* src, size_t num_bytes) {
  allocator_->memcpy(dest, src, num_bytes);
}

int64_t Device::get_sys_thread_count() {
  static int64_t sys_thread_count = 0;

  if (0 >= sys_thread_count) {
#ifdef __GNUC__
    sys_thread_count = static_cast<int>(sysconf(_SC_NPROCESSORS_CONF));
#elif defined(_MSC_VER)
    SYSTEM_INFO info;
    GetSystemInfo(&info);
    sys_thread_count = static_cast<int>(info.dwNumberOfProcessors);
#endif

    if (0 >= sys_thread_count) {
      sys_thread_count = 4;
    }
  }

  return sys_thread_count;
}

std::shared_ptr<Device> Device::default_device() {
  static auto device = std::make_shared<Device>(DeviceType::CPU, 0);

  return device;
}

}