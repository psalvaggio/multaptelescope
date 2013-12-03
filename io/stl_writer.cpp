// File Description
// Author: Philip Salvaggio

#include "stl_writer.h"
#include <cstdio>

StlWriter::StlWriter(const std::string& filename)
    : output_file_(NULL),
      num_triangles_(0) {
  output_file_ = fopen(filename.c_str(), "wb");

  if (is_open()) {
    const int kHeaderSize = 84;

    unsigned char header[kHeaderSize];
    for (int i = 0; i < kHeaderSize; i++) {
      header[i] = 0;
    }
    fwrite(header, kHeaderSize, 1, output_file_);
  }
}

StlWriter::~StlWriter() {
  if (is_open()) {
    fseek(output_file_, 80, SEEK_SET);
    fwrite(&num_triangles_, sizeof(uint32_t), 1, output_file_);
    fclose(output_file_);
  }
}

void StlWriter::AddTriangle(const float* vertices) {
  if (!vertices || !is_open()) return;
  num_triangles_++;

  float side1[3], side2[3];
  side1[0] = vertices[3] - vertices[0];
  side1[1] = vertices[4] - vertices[1];
  side1[2] = vertices[5] - vertices[2];
  side2[0] = vertices[6] - vertices[0];
  side2[1] = vertices[7] - vertices[1];
  side2[2] = vertices[8] - vertices[2];

  float triangle[12];
  triangle[0] = side1[1]*side2[2] - side1[2]*side2[1];
  triangle[1] = side1[2]*side2[0] - side1[0]*side2[2];
  triangle[2] = side1[0]*side2[1] - side1[1]*side2[0];
  for (int i = 0; i < 9; i++) {
    triangle[i+3] = vertices[i];
  }

  fwrite(triangle, sizeof(float), 12, output_file_);

  uint16_t attribute = 0;
  fwrite(&attribute, sizeof(uint16_t), 1, output_file_);
}
