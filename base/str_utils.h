// File Description
// Author: Philip Salvaggio

#ifndef STR_UTILS_H
#define STR_UTILS_H

#include <algorithm> 
#include <functional> 
#include <cctype>
#include <cstdarg>
#include <locale>
#include <string>
#include <vector>
#include <sstream>

namespace mats {

// Trim whitespace from the start.
inline std::string& ltrim(std::string& s) {
  s.erase(s.begin(),
          std::find_if(s.begin(),
                       s.end(),
                       std::not1(std::ptr_fun<int, int>(std::isspace))));
  return s;
}

// Trim whitespace from the end.
inline std::string& rtrim(std::string& s) {
  s.erase(std::find_if(s.rbegin(), s.rend(),
  std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
  return s;
}

// Trim whitespace from both ends.
inline std::string& trim(std::string& s) {
  return ltrim(rtrim(s));
}

// Equivalent of PHP's explode() function. Splits a string on a given delimiter.
void explode(const std::string& s,
             char delim,
             std::vector<std::string>* result);
void explode(const std::string& s,
             std::string regex_str,
             std::vector<std::string>* result);

// Equivalent of PHP's implode() function.
std::string implode(const std::vector<std::string>& parts, std::string delim);

void StringAppendf(std::string* output, const char* format, va_list vargs);

std::string StringPrintf(const char* format, ...);

void SStringPrintf(std::string* output, const char* format, ...);

bool starts_with(const std::string& haystack, const std::string& needle);
bool ends_with(const std::string& haystack, const std::string& needle);

}

#endif  // STR_UTILS_H
