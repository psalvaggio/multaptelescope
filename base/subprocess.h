// A class for dealing with subprocesses.
// Author: Philip Salvaggio

#ifndef SUBPROCESS_H
#define SUBPROCESS_H

#include <string>

class Subprocess {
 public:
  static std::string exec(const char* cmd);
};

#endif
