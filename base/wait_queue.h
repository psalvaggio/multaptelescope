// File Description
// Author: Philip Salvaggio

#ifndef WAIT_QUEUE_H
#define WAIT_QUEUE_H

#include <condition_variable>
#include <mutex>
#include <queue>
#include <memory>

template<typename T>
class WaitQueue {
 public:
  // Constructor
  //
  // Parameters:
  //  capactity - The maximum capacity of the queue. If -1, then the queue has
  //              no upper bound. Otherwise, elements beyond the capacity will
  //              not be inserted.
  explicit WaitQueue(int capacity = -1);

  WaitQueue(WaitQueue&& other);
  WaitQueue(const WaitQueue& other) = delete;

  WaitQueue& operator=(WaitQueue&& other);

  int capacity() const { return capacity_; }

  template<typename... Args>
  void emplace(Args&&... args);

  bool empty() const { return queue_.empty(); }

  // Push a value onto the queue. T must be copy-constructable.
  void push(const T& val);

  // Push a value onto the queue. The queue assumes ownership of val.
  void push(T* val);

  size_t size() const { return queue_.size(); }

  T* wait();

  T* wait_for(size_t msec);

 private:
  void pop();

 private:
  int capacity_;
  std::queue<std::unique_ptr<T>> queue_;
  std::mutex mutex_;
  std::condition_variable new_image_event_;
  std::mutex new_image_mutex_;
};

#include "wait_queue.hpp"

#endif  // WAIT_QUEUE_H
