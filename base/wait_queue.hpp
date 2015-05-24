// File Description
// Author: Philip Salvaggio

#ifndef WAIT_QUEUE_HPP
#define WAIT_QUEUE_HPP

#include "wait_queue.h"

#include <iostream>

template<typename T>
WaitQueue<T>::WaitQueue(int capacity)
    : capacity_(capacity),
      queue_(),
      mutex_(),
      new_image_event_(),
      new_image_mutex_() {}

template<typename T>
WaitQueue<T>::WaitQueue(WaitQueue<T>&& other)
    : capacity_(other.capacity_),
      queue_(std::move(other.queue_)),
      mutex_(),
      new_image_event_(),
      new_image_mutex_() {}

template<typename T>
WaitQueue<T>& WaitQueue<T>::operator=(WaitQueue<T>&& other) {
  capacity_ = other.capacity_;
  queue_ = std::move(other.queue_);
}

template<typename T>
template<typename... Args>
void WaitQueue<T>::emplace(Args&&... args) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (size() < capacity_ || capacity_ == -1) {
    queue_.emplace(new T(std::forward<Args>(args)...));

    std::unique_lock<std::mutex> mtx_lock(new_image_mutex_);
    new_image_event_.notify_one();
  }
}

template<typename T>
void WaitQueue<T>::pop() {
  std::lock_guard<std::mutex> lock(mutex_);
  queue_.pop();
}

template<typename T>
void WaitQueue<T>::push(const T& val) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (size() < capacity_ || capacity_ == -1) {
    queue_.push(new T(val));

    std::unique_lock<std::mutex> mtx_lock(new_image_mutex_);
    new_image_event_.notify_one();
  }
}

template<typename T>
void WaitQueue<T>::push(T* val) {
  std::lock_guard<std::mutex> lock(mutex_);
  std::unique_ptr<T> ptr(val);
  if (static_cast<int>(size()) < capacity_ || capacity_ == -1) {
    queue_.emplace(std::move(ptr));

    std::unique_lock<std::mutex> mtx_lock(new_image_mutex_);
    new_image_event_.notify_one();
  }
}

template<typename T>
std::unique_ptr<T> WaitQueue<T>::wait() {
  std::unique_lock<std::mutex> lock(mutex_);

  if (empty()) {
    lock.unlock();
    std::unique_lock<std::mutex> mtx_lock(new_image_mutex_);
    new_image_event_.wait(mtx_lock);
    lock.lock();
  }

  std::unique_ptr<T> ptr(std::move(queue_.front()));
  queue_.pop();
  return std::move(ptr);
}

template<typename T>
std::unique_ptr<T> WaitQueue<T>::wait_for(size_t msec) {
  std::unique_lock<std::mutex> lock(mutex_);

  if (empty()) {
    lock.unlock();
    std::unique_lock<std::mutex> mtx_lock(new_image_mutex_);
    auto status = new_image_event_.wait_for(mtx_lock,
        std::chrono::milliseconds(msec));
    if (status == std::cv_status::timeout)  return nullptr;
    lock.lock();
  }

  std::unique_ptr<T> ptr(std::move(queue_.front()));
  queue_.pop();
  return std::move(ptr);
}


#endif  // WAIT_QUEUE_HPP
