// File Description
// Author: Philip Salvaggio

#ifndef MACROS_H
#define MACROS_H

#define NO_COPY_OR_ASSIGN(class_name) \
 private: \
  class_name(const class_name& other); \
  class_name& operator=(const class_name& other);

#define NO_CONSTRUCTION(class_name) \
 private: \
  class_name(); \
  class_name(const class_name& other); \
  class_name& operator=(const class_name& other);

#define SINGLETON(class_name) \
 public: \
  static class_name& getInstance() { \
    static class_name instance; \
    return instance; \
  } \
 private: \
  class_name(); \
  class_name(const class_name& other); \
  class_name& operator=(const class_name& other);

#endif  // MACROS_H
