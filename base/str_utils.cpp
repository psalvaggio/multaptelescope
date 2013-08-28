// File Description
// Author: Philip Salvaggio

#include "str_utils.h"

namespace mats {

// Equivalent of PHP's explode() function. Splits a string on a given delimiter.
void explode(const std::string& s,
             char delim,
             std::vector<std::string>* result) {
  std::istringstream iss(s);

  for (std::string token; std::getline(iss, token, delim); ) {
    result->push_back(token);
  }
}

void StringAppendf(std::string* output, const char* format, va_list vargs) {
  int size = 1024;
  char* buffer = NULL;
  int length = 0;

  for (;;) {
    buffer = new char[size];

    va_list tmp_vargs;
    va_copy(tmp_vargs, vargs);
    length = vsnprintf(buffer, size, format, tmp_vargs);
    va_end(tmp_vargs);

    if (length >= 0 && length < size) {
      break;
    }

    delete[] buffer;

    if (length >= size) {
      size = length + 1;
    } else {
      return;
    }
  } 

  output->append(buffer, length);
  delete[] buffer;
}

std::string StringPrintf(const char* format, ...) {
  va_list vargs;
  va_start(vargs, format);
  std::string output;
  StringAppendf(&output, format, vargs);
  va_end(vargs);
  return output;
}

void SStringPrintf(std::string* output, const char* format, ...) {
  va_list vargs;                                   
  va_start(vargs, format);
  StringAppendf(output, format, vargs);
  va_end(vargs);
}

}
