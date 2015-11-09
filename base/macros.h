// File Description
// Author: Philip Salvaggio

#ifndef MACROS_H
#define MACROS_H

#define SINGLETON(class_name) \
 public: \
  class_name(const class_name& other) = delete; \
  class_name& operator=(const class_name& other) = delete; \
  \
  static class_name& getInstance() { \
    static class_name instance; \
    return instance; \
  } \
 private: \
  class_name();

#endif  // MACROS_H
