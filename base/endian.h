// File Description
// Author: Philip Salvaggio

#ifndef ENDIAN_H
#define ENDIAN_H

#include <cstddef>

namespace mats {

// Endian-swapping routine. From:
// http://stackoverflow.com/questions/105252/
//   how-do-i-convert-between-big-endian-and-little-endian-values-in-c
template <typename T> T swap_endian(T u) {
  union {
    T u;
    unsigned char u8[sizeof(T)];
  } source, dest;

  source.u = u;

  for (size_t k = 0; k < sizeof(T); k++)
    dest.u8[k] = source.u8[sizeof(T) - k - 1];

  return dest.u;
}

// Detect whether the system is little Endian
inline bool isLittleEndian() {
  short int number = 0x1;
  char *numPtr = (char*)&number;
  return (numPtr[0] == 1);
}

}

#endif  // ENDIAN_H
