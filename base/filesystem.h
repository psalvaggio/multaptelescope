// File Description
// Author: Philip Salvaggio

#ifndef FILESYSTEM_H
#define FILESYSTEM_H

#include <string>

namespace mats {

bool is_dir(const std::string& path);
bool file_exists(const std::string& path);

}

#endif  // FILESYSTEM_H
