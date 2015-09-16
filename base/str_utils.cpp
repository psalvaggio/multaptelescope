// File Description
// Author: Philip Salvaggio

#include "str_utils.h"

#include <cstdio>

#if __GNUC__ == 4 && __GNUC_MINOR__ < 9
  #include <boost/regex.hpp>
  using boost::sregex_token_iterator;
  using boost::regex;
#else
  #include <regex>
#endif

using namespace std;

namespace mats {

// Equivalent of PHP's explode() function. Splits a string on a given delimiter.
void explode(const string& s, char delim, vector<string>* result) {
  istringstream iss(s);

  for (string token; getline(iss, token, delim); ) {
    result->push_back(token);
  }
}

void explode(const string& s, string regex_str, vector<string>* result) {
  if (!result) return;
  result->clear();
  regex delim(regex_str);

  sregex_token_iterator srit(begin(s), end(s), delim, -1);
  sregex_token_iterator srend;
  while (srit != srend) {
    result->push_back(*srit);
    ++srit;
  }
}

void StringAppendf(string* output, const char* format, va_list vargs) {
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

string StringPrintf(const char* format, ...) {
  va_list vargs;
  va_start(vargs, format);
  string output;
  StringAppendf(&output, format, vargs);
  va_end(vargs);
  return output;
}

void SStringPrintf(string* output, const char* format, ...) {
  va_list vargs;                                   
  va_start(vargs, format);
  StringAppendf(output, format, vargs);
  va_end(vargs);
}

bool starts_with(const string& haystack, const string& needle) {
  if (haystack.length() >= needle.length()) {
    return haystack.compare(0, needle.length(), needle) == 0;
  }
  return false;
}

bool ends_with(const string& haystack, const string& needle) {
  if (haystack.length() >= needle.length()) {
    return haystack.compare(haystack.length() - needle.length(),
                            needle.length(),
                            needle) == 0;
  }
  return false;
}

string AppendSlash(const string& input) {
  if (input.back() != '/') {
    return input + '/';
  }
  return input;
}

}
