#ifndef EXCEPTION_H
#define EXCEPTION_H

namespace dlfm {

#define ARGUMENT_CHECK(condition, message)                                     \
  if (!(condition)) {                                                          \
    std::ostringstream oss;                                                    \
    oss << __FILE__ << ":";                                                    \
    oss << __LINE__ << ":";                                                    \
    oss << message << ".";                                                     \
    throw std::invalid_argument(oss.str());                                    \
  }

#define RUNTIME_ERROR(message)                                                 \
  {                                                                            \
    std::ostringstream oss;                                                    \
    oss << __FILE__ << ":";                                                    \
    oss << __LINE__ << ":";                                                    \
    oss << message << ".";                                                     \
    throw std::runtime_error(oss.str());                                       \
  }

}

#endif