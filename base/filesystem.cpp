// File Description
// Author: Philip Salvaggio

#include "filesystem.h"

#include "base/str_utils.h"

#include <sys/stat.h>
#include <dirent.h>

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

void scandir(const std::string& path,
             const std::string& extension,
             std::vector<std::string>* files) {
  files->clear();

  DIR* dir = opendir(path.c_str());
  if (!dir) return;

  struct dirent entry;
  struct dirent* result;

  while (readdir_r(dir, &entry, &result) == 0) {
    if (result == NULL) break;
    if (mats::ends_with(entry.d_name, extension)) {
      files->emplace_back(entry.d_name);
    }
  }

  closedir(dir);
}

}
