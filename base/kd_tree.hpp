// A limited kd-tree implementation. I needed something simple and didn't want
// to jump through hoops to use FLANN.
// Author: Philip Salvaggio

#ifndef KD_TREE_HPP_
#define KD_TREE_HPP_

#include "kd_tree.h"

#include <cmath>
#include <numeric>

template<typename T, int Dim>
KDTree<T, Dim>::KDTree()
    : kd_tree_(), data_(), size_(0) {
  NeighborComp = [] (const neighbor_t& a, const neighbor_t& b) {
      return a.second < b.second;
    };
}


template<typename T, int Dim>
void KDTree<T, Dim>::push_back(const T& element) {
  data_.push_back(element);
}


template<typename T, int Dim>
template<typename ... Args>
void KDTree<T, Dim>::emplace_back(Args&&... args) {
  data_.emplace_back(std::forward<Args>(args)...);
}


template<typename T, int Dim>
void KDTree<T, Dim>::build() {
  kd_tree_.clear();
  size_ = data_.size();

  size_t size = 1;
  while (size - 1 < size_) size = size << 1;
  size = (size << 1) - 1;
  kd_tree_.resize(size);
  for (auto& tmp : kd_tree_) {
    tmp.first = -1;
    tmp.second = 0;
  }

  std::vector<int> indices(data_.size(), 0);
  std::iota(indices.begin(), indices.end(), 0);
  BuildKdTree(0, indices);
}


template<typename T, int Dim>
void KDTree<T, Dim>::clear() {
  kd_tree_.clear();
  data_.clear();
}


template<typename T, int Dim>
void KDTree<T, Dim>::kNNSearch(const T& query,
                               int k,
                               double max_radius,
                               std::vector<int>* results) const {
  results->clear();
  std::vector<neighbor_t> idx_results;
  double max_dist2 = max_radius * max_radius;

  size_t max_neighbors = k < 0 ? data_.size() : k;
  kNNSearchHelper(query, max_neighbors, 0, max_dist2, &idx_results);

  for (const auto& idx : idx_results) {
    results->push_back(idx.first);
  }
}


template<typename T, int Dim>
void KDTree<T, Dim>::BuildKdTree(size_t root_idx, std::vector<int>& indices) {
  std::vector<double> min_dims(Dim, std::numeric_limits<double>::max()),
                      max_dims(Dim, std::numeric_limits<double>::min());

  for (size_t i = 0; i < indices.size(); i++) {
    for (int j = 0; j < Dim; j++) {
      double val = static_cast<double>(data_[indices[i]][j]);
      min_dims[j] = std::min(min_dims[j], val);
      max_dims[j] = std::max(max_dims[j], val);
    }
  }

  int dimension = 0;
  double max_diff = 0;
  for (int i = 0; i < Dim; i++) {
    double diff = max_dims[i] - min_dims[i];
    if (diff > max_diff) {
      dimension = i;
      max_diff = diff;
    }
  }

  std::sort(indices.begin(), indices.end(),
      [this, dimension] (int a, int b) {
        return data_[a][dimension] < data_[b][dimension];
      });
  size_t median_idx = indices.size() / 2;

  kd_tree_[root_idx].first = indices[median_idx];
  kd_tree_[root_idx].second = dimension;

  if (median_idx > 0) {
    std::vector<int> less_than;
    less_than.insert(less_than.end(),
                     indices.begin(), indices.begin() + median_idx);
    BuildKdTree(2 * root_idx + 1, less_than);
  }
  if (median_idx < indices.size() - 1) {
    std::vector<int> greater_than;
    greater_than.insert(greater_than.end(),
                        indices.begin() + (median_idx+1), indices.end());
    BuildKdTree(2 * (root_idx + 1), greater_than);
  }
}


template<typename T, int Dim>
void KDTree<T, Dim>::kNNSearchHelper(
    const T& query,
    size_t k,
    size_t examine_index,
    double &max_distance_sq,
    std::vector<neighbor_t>* results) const {
  if (examine_index >= kd_tree_.size()) return;

  int index = kd_tree_[examine_index].first;
  if (index < 0 || index >= int(data_.size())) return;
  const T& neighbor = data_[index];

  size_t left_child = 2*examine_index+1;
  size_t right_child = left_child+1;

  int dim = kd_tree_[examine_index].second;
  double dim_diff = query[dim] - neighbor[dim];

  if (dim_diff < 0) {
    kNNSearchHelper(query, k, left_child, max_distance_sq, results);

    if (dim_diff*dim_diff < max_distance_sq || results->size() < k) {
      kNNSearchHelper(query, k, right_child, max_distance_sq, results);
    }
  } else {
    kNNSearchHelper(query, k, right_child, max_distance_sq, results);

    if (dim_diff*dim_diff < max_distance_sq || results->size() < k) {
      kNNSearchHelper(query, k, left_child, max_distance_sq, results);
    }
  }
 
  double dist2 = 0;
  for (int i = 0; i < Dim; i++) {
    dist2 += std::pow(neighbor[i] - query[i], 2);
  }

  if (dist2 > max_distance_sq) return;

  if (results->size() < k) {
    results->emplace_back(index, dist2);

    if (results->size() == k) {
      std::make_heap(results->begin(), results->end(), NeighborComp);
    }
  } else if (dist2 < results->front().second) {
    results->emplace_back(index, dist2);
    std::push_heap(results->begin(), results->end(), NeighborComp);
    std::pop_heap(results->begin(), results->end(), NeighborComp);
    results->pop_back();
  }
}

#endif
