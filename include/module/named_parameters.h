#include "common/basic.h"

namespace dlfm::nn {

// A common parmater type.
class ParameterType {
private:
  std::string key_;

  uint64_t has_val_;

  size_t number_val_;
  std::vector<size_t> array_val_;

public:
  ParameterType(const std::string &key);

  const std::string& key();

  bool flag_val() const;
  bool has_flag_val() const;

  size_t number_val() const;
  bool has_number_val() const;

  const std::vector<size_t>& array_val() const;
  bool has_array_val() const;

  void number_val(size_t number_val);
  void array_val(const std::vector<size_t> &array_val);
};

struct NamedParameterIndicator {
  std::string key;

  NamedParameterIndicator(const std::string &k);

  // Override operator = to accept different value.
  ParameterType operator=(size_t number_val) const;

  ParameterType operator=(const std::vector<size_t> &array_val) const;
};

// fetch parameters.
template <typename Type>
void fetch_variadic_args(std::unordered_map<std::string, Type> &maps) {
  // nothing.
}

template <typename Type, typename...Args>
void fetch_variadic_args(std::unordered_map<std::string, Type> &maps, Type head, Args... args) {
  maps.insert(std::make_pair(head.key(), head));

  fetch_variadic_args(maps, args...);
}

extern const NamedParameterIndicator kernel_size;
extern const NamedParameterIndicator stride;
extern const NamedParameterIndicator padding;
extern const NamedParameterIndicator dilation;
extern const NamedParameterIndicator groups;
extern const NamedParameterIndicator bias;

}