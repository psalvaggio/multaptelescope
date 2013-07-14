// Copyright 2011, Digital Imaging and Remote Sensing Laboratory
// All Rights Reserverd.
// Written by Philip Salvaggio (pss7119@rit.edu)
//
// A scoped pointer is a simple implementation of a smart pointer in that it
// deletes it's pointer as soon as it goes out of scope. It can be used
// exactly like a normal pointer with the overloaded operators. This file also
// defines scoped_array for array pointers.

#ifndef BASE_ASSERTIONS_H_
#define BASE_ASSERTIONS_H_

#include <iostream>
#include <cassert>

#define CHECK(expr) \
  if (!(expr)) { \
    std::cerr << "CHECK failed: " << #expr << std::endl; \
    assert(expr); \
  }

#endif  // BASE_ASSERTIONS_H_
