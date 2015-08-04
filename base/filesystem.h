// File Description
// Author: Philip Salvaggio

#ifndef FILESYSTEM_H
#define FILESYSTEM_H

#include <string>
#include <vector>

namespace mats {

bool is_dir(const std::string& path);
bool file_exists(const std::string& path);
void scandir(const std::string& path,
             const std::string& extension,
             std::vector<std::string>* files);
std::string ResolvePath(const std::string& path);

}

#endif  // FILESYSTEM_H
