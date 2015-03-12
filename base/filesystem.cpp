// File Description
// Author: Philip Salvaggio

#include "filesystem.h"
#include <sys/stat.h>

namespace mats {

bool is_dir(const std::string& path) {
  struct stat buf;
  stat(path.c_str(), &buf);
  return S_ISDIR(buf.st_mode);
}

bool file_exists(const std::string& path) {
  struct stat buffer;   
  return stat(path.c_str(), &buffer) == 0;
}

}
