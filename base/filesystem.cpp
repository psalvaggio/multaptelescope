// File Description
// Author: Philip Salvaggio

#include "filesystem.h"

#include "base/str_utils.h"

#include <sys/stat.h>
#include <dirent.h>
#include <wordexp.h>

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

void subdirectories(const std::string& path,
                    std::vector<std::string>* subdirs) {
  subdirs->clear();

  std::string dir = AppendSlash(path);
  std::vector<std::string> files;
  scandir(dir, "", &files);
  for (const auto& file : files) {
    if (file == "." || file == "..") continue;

    auto subpath = dir + file;
    if (is_dir(subpath)) {
      subdirs->push_back(file);
    }
  }
}

std::string ResolvePath(const std::string& path) {
  wordexp_t exp_result;
  wordexp(path.c_str(), &exp_result, 0);
  std::string new_path = exp_result.we_wordv[0];
  wordfree(&exp_result);

  if (is_dir(new_path)) {
    new_path = mats::AppendSlash(new_path);
  }

  return new_path;
}

std::string Basename(const std::string& path,
                     const std::string& extension) {
  size_t found = path.find_last_of("/\\");
  std::string basename = path;
  if (found != std::string::npos) {
    basename = path.substr(found + 1);
  }
  if (extension.size() > 0 && ends_with(basename, extension)) {
    basename = basename.substr(0, basename.size() - extension.size());
  }
  return basename;
}

}
