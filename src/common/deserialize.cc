#include "common/exception.h"
#include "common/deserialize.h"

namespace dlfm {

FileStreamReader::FileStreamReader(const std::string &file_path)
:file_path_(file_path) {
  fstream_ = std::ifstream(file_path, std::fstream::binary | std::fstream::in);
  ARGUMENT_CHECK(fstream_.is_open(), "file:" << file_path_ << " open error!");
}

FileStreamReader::~FileStreamReader() {
  fstream_.close();
}

bool FileStreamReader::read(void *byte, size_t length) {
  if (!fstream_.is_open()) {
    return false;
  }

  fstream_.read((char*)byte, length);

  return true;
}

ModelDeserialize::ModelDeserialize(const std::string &file_path): reader_(file_path) {
}

}