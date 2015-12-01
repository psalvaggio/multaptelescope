// A limited kd-tree implementation. I needed something simple and didn't want
// to jump through hoops to use FLANN.
// Author: Philip Salvaggio

#ifndef KD_TREE_H_
#define KD_TREE_H_

#include <functional>
#include <vector>

template<typename T, int Dim>
class KDTree {
 public:
  using Distance = std::function<double(const T&, const T&)>;

  KDTree();

  KDTree(const KDTree<T, Dim>& other) = delete;
  KDTree<T, Dim>& operator=(const KDTree<T, Dim>& other) = delete;

  void push_back(const T& element);

  template<typename ... Args>
  void emplace_back(Args&&... args);

  T& back() { return data_.back(); }
  T& operator[](int index) { return data_[index]; }
  const T& operator[](int index) const { return data_[index]; }

  void build();

  void clear();

  size_t size() const { return size_; }

  template<typename Query>
  void kNNSearch(const Query& query,
                 int k,
                 double max_radius,
                 std::vector<int>* results) const;

 private:
  void BuildKdTree(size_t root_idx, std::vector<int>& indices);

  using neighbor_t = std::pair<int, double>;
  template<typename Query>
  void kNNSearchHelper(const Query& query,
                       size_t k,
                       size_t examine_index,
                       double &max_distance,
                       std::vector<neighbor_t>* results) const;

  void TestAndInsertIntoNearestQueue(const T& query,
                                     T* photon,
                                     std::vector<T>& nearest,
                                     size_t max_photons,
                                     double max_dist_sq) const;

 private:
  std::vector<std::pair<int, int>> kd_tree_;
  std::vector<T> data_;
  size_t size_;
  std::function<bool(const neighbor_t& a, const neighbor_t& b)> NeighborComp;
};

#include "kd_tree.hpp"

#endif
