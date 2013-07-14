// Copyright 2011, Digital Imaging and Remote Sensing Laboratory
// All Rights Reserverd.
// Written by Philip Salvaggio (pss7119@rit.edu)
//
// A scoped pointer is a simple implementation of a smart pointer in that it
// deletes it's pointer as soon as it goes out of scope. It can be used
// exactly like a normal pointer with the overloaded operators. This file also
// defines scoped_array for array pointers.

#ifndef BASE_SCOPED_PTR_H_
#define BASE_SCOPED_PTR_H_

#include <cstddef>
#include "assertions.h"

template <class T>
class scoped_ptr {
 public:
  scoped_ptr() : ptr_(NULL) {}
  explicit scoped_ptr(T* ptr) : ptr_(ptr) {};

  virtual ~scoped_ptr() { Clear(); }
  virtual void Clear() {
    if (ptr_ != NULL) {
      delete ptr_;
      ptr_ = NULL;
    }
  }

  void Reset(T* new_ptr) {
    Clear();
    ptr_ = new_ptr;
  }

  T* get() { return ptr_; }
  const T* get() const { return ptr_; }

  T* Detach() {
    T* ptr = ptr_;
    ptr_ = NULL;
    return ptr;
  }

  T& operator*() { return *ptr_; }
  const T& operator*() const { return *ptr_; }
  T* operator->() { CHECK(ptr_); return ptr_; }
  const T* operator->() const { return ptr_; }

  bool IsNull() const { return !ptr_; }

 protected:
  T* ptr_;
};

template <class T>
class scoped_array : public scoped_ptr<T> {
 public:
  scoped_array() {}
  explicit scoped_array(T* ptr) : scoped_ptr<T>(ptr) {}

  virtual ~scoped_array() { Clear(); }
  virtual void Clear() {
    if (this->ptr_ != NULL) {
      delete[] this->ptr_;
      this->ptr_ = NULL;
    }
  }
};

#endif  // BASE_SCOPED_PTR_H_
