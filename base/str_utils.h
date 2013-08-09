// File Description
// Author: Philip Salvaggio

#ifndef STR_UTILS_H
#define STR_UTILS_H

#include <algorithm> 
#include <functional> 
#include <cctype>
#include <locale>

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
             std::vector<std::string>* result) {
  std::istringstream iss(s);

  for (std::string token; std::getline(iss, token, delim); ) {
    result->push_back(token);
  }
}

}

#endif  // STR_UTILS_H
