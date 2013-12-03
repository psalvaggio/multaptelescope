// File Description
// Author: Philip Salvaggio

#ifndef STL_WRITER_H
#define STL_WRITER_H

#include <cstdio>
#include <string>

class StlWriter {
 public:
  StlWriter(const std::string& filename);
  ~StlWriter();

  bool is_open() const { return output_file_ != NULL; }

  void AddTriangle(const float* vertices);

 private:
  FILE* output_file_;
  uint32_t num_triangles_;
};

#endif  // STL_WRITER_H
