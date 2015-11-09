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

void subdirectories(const std::string& path,
                    std::vector<std::string>* subdirs);

std::string AppendSlash(const std::string& input);

std::string ResolvePath(const std::string& path);

std::string Basename(const std::string& path,
                     const std::string& extension = "");

std::string DirectoryName(const std::string& path);

}

#endif  // FILESYSTEM_H
