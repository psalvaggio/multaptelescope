// File Description
// Author: Philip Salvaggio

#include "usaf_1951_target.h"

#include "base/opencv_utils.h"

#include <iostream>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

Usaf1951Target::Usaf1951Target(const Mat& image, int num_levels)
    : image_(), num_levels_(num_levels), bounding_boxes_(), mean_vectors_() {
  image.copyTo(image_);
}

bool Usaf1951Target::RecognizeTarget() {
  Mat image;
  image_.copyTo(image);

  // Convert to an 8-bit image
  switch (image.type()) {
    case CV_8UC1: break;
    case CV_16SC1:
      image.convertTo(image, CV_8UC1, 1./256);
      break;
    case CV_32SC1:
      image.convertTo(image, CV_8UC1, 1./65536);
      break;
    default:
      cerr << "Error: Unsupported image data type." << endl;
      return false;
  }

  // Threshold the image with Otsu's method
  Mat image_bw;
  threshold(image, image_bw, 0, 1, THRESH_BINARY | THRESH_OTSU);

  // Perfom connected components analysis. cc_labels is an image where each
  // pixel value is the component label. cc_centroids is a 2d array where the
  // centroid of component i is cc_centroids(i, 0) and cc_centroids(i, 1).
  Mat_<int32_t> cc_labels, cc_stats;
  Mat_<double> cc_centroids;
  int num_ccs = connectedComponentsWithStats(
      image_bw, cc_labels, cc_stats, cc_centroids, 8, CV_32S);

  // The aspect ratio on the bars in the USAF target is 5:1. Since we aren't
  // guaranteed that the bars are aligned with the pixel grid, we need to
  // perform principal components on each component and ratio the square roots
  // of the eigenvalues to get the aspect ratio. bar_ccs is a list of label
  // indices that are bars.
  vector<int> bar_ccs;
  vector<Vector2d> bar_orientations;
  DetermineBars(cc_labels, cc_stats, cc_centroids, num_ccs,
                &bar_ccs, &bar_orientations);

  // The USAF target has both horizontal and vertical bars. Filter the bars into
  // groups based on angle difference in the principal direction.
  vector<Vector2d> mean_vectors;
  vector<vector<int>> oriented_bars;
  SplitHorizontalVerticalBars(bar_ccs, bar_orientations,
                              &oriented_bars, &mean_vectors);

  if (oriented_bars.size() < 2) {
    cerr << "Error: Only one orientation of bars found." << endl;
    return false;
  }

  // For each orientation, detect groups of tri-bars and sort them in decreasing
  // bar size.
  vector<vector<TriBar>> bar_groups;
  for (size_t i = 0; i < oriented_bars.size(); i++) {
    bar_groups.emplace_back();
    DetectTriBars(oriented_bars[i], mean_vectors[i], cc_centroids,
                  &(bar_groups.back()));

    sort(begin(bar_groups.back()), end(bar_groups.back()),
        [&cc_stats] (const TriBar& a, const TriBar& b) {
          return cc_stats(get<0>(a), CC_STAT_AREA) >
                 cc_stats(get<0>(b), CC_STAT_AREA);
        });
  }

  DetectMisses(bar_groups, cc_stats);

  bounding_boxes_.clear();
  vector<vector<Vector2d>> bb_centroids;
  FindBoundingBoxes(bar_groups, cc_labels, cc_stats, mean_vectors,
                    &bounding_boxes_, &bb_centroids);

  CompletePartialPairs(bounding_boxes_, bb_centroids);

  for (int i = 1; i < num_levels_; i++) {
    CompleteLowerLevel(bounding_boxes_, bb_centroids, i);
  }

  return true;
}


void Usaf1951Target::GetProfile(int/*, something*/) {
}


Mat Usaf1951Target::VisualizeBoundingBoxes() const {
  vector<Scalar> colors {Scalar(0, 0, 1),
                         Scalar(0, 1, 0),
                         Scalar(1, 0, 0),
                         Scalar(1, 1, 0),
                         Scalar(1, 0, 1),
                         Scalar(0, 1, 1),
                         Scalar(1, 1, 1),
                         Scalar(0, 0.5, 1),
                         Scalar(0.5, 0, 1),
                         Scalar(0.65, 0.65, 0.65),
                         Scalar(0.85, 0.65, 0),
                         Scalar(0.35, 0.95, 0.35)};

  Mat bars(image_.size(), CV_32FC3);
  cvtColor(image_, bars, COLOR_GRAY2RGB);
  for (size_t i = 0; i < bounding_boxes_.size(); i++) {
    for (size_t j = 0; j < bounding_boxes_[i].size(); j++) {
      if (bounding_boxes_[i][j][0] != 0) {
        size_t size = bounding_boxes_[i][j].size();
        for (size_t k = 0; k < size; k += 2) {
          line(bars,
               Point(bounding_boxes_[i][j][k], bounding_boxes_[i][j][k+1]),
               Point(bounding_boxes_[i][j][(k+2)%size],
                     bounding_boxes_[i][j][(k+3)%size]),
               255 * colors[j % colors.size()]);
        }
      }
    }
  }

  return bars;
}


void Usaf1951Target::PcaAnalysis(const Mat_<int32_t>& cc_labels,
                                 const Mat_<double>& cc_centroids,
                                 int label,
                                 vector<Pca2dResult>* results) const {
  Mat_<int16_t> bar = cc_labels == label;
  double mean_x = cc_centroids(label, 0),
         mean_y = cc_centroids(label, 1);

  Mat_<double> cov(2, 2);
  cov = 0;
  int count = 0;
  for (int y = 0; y < bar.rows; y++) {
    double delta_y = y - mean_y;
    for (int x = 0; x < bar.cols; x++) {
      if (bar(y, x) > 0) {
        double delta_x = x - mean_x;
        cov(0, 0) += delta_x * delta_x;
        cov(0, 1) += delta_x * delta_y;
        cov(1, 0) += delta_y * delta_x;
        cov(1, 1) += delta_y * delta_y;
        count++;
      }
    }
  }
  cov /= count;

  Mat_<double> eigval, eigvec;
  eigen(cov, eigval, eigvec);

  results->emplace_back(eigval(0), eigvec(0, 0), eigvec(0, 1));
  results->emplace_back(eigval(1), eigvec(1, 0), eigvec(1, 1));
}


void Usaf1951Target::DetermineBars(
    const Mat_<int32_t>& cc_labels,
    const Mat_<int32_t>& cc_stats,
    const Mat_<double>& cc_centroids,
    int num_ccs,
    vector<int>* bar_ccs,
    vector<Vector2d>* bar_orientations) const {
  const double kUsafBarAspect = 0.2;
  const double kUsafBarAspectTolerance = 0.05;

  for (int i = 1; i < num_ccs; i++) {
    int pixel_area = cc_stats(i, CC_STAT_AREA);
    if (pixel_area < 5) continue;

    vector<tuple<double, double, double>> pca;
    PcaAnalysis(cc_labels, cc_centroids, i, &pca);

    double aspect = sqrt(get<0>(pca[0])) / sqrt(get<0>(pca[1]));

    if (abs(aspect - kUsafBarAspect) > kUsafBarAspectTolerance &&
        abs(1 / aspect - kUsafBarAspect) > kUsafBarAspectTolerance) {
      continue;
    }

    double vec_x = get<1>(pca[0]),
           vec_y = get<2>(pca[0]);
    if ((abs(vec_x) > abs(vec_y) && vec_x < 0) ||
        (abs(vec_y) > abs(vec_x) && vec_y < 0)) {
      vec_x *= -1; vec_y *= -1;
    }
    double mag = sqrt(vec_x*vec_x + vec_y*vec_y);
    bar_ccs->push_back(i);
    bar_orientations->emplace_back(vec_x / mag, vec_y / mag);
  }
}


void Usaf1951Target::SplitHorizontalVerticalBars(
    const vector<int>& bar_ccs,
    const vector<Vector2d>& bar_orientations,
    vector<vector<int>>* oriented_bars,
    vector<Vector2d>* mean_vectors) {
  const double kAngleTolerance = 5 * M_PI / 180;
  for (size_t i = 0; i < bar_ccs.size(); i++) {
    int insert_idx = -1;
    double max_dot = 0;

    for (size_t j = 0; j < oriented_bars->size(); j++) {
      double dot = get<0>(bar_orientations[i]) * get<0>((*mean_vectors)[j]) +
                   get<1>(bar_orientations[i]) * get<1>((*mean_vectors)[j]);
      if (abs(dot) > abs(max_dot)) {
        max_dot = dot;
        if (acos(abs(dot)) < kAngleTolerance) {
          insert_idx = j;
        }
      }
    }

    double refl = max_dot < 0 ? -1 : 1;
    if (insert_idx == -1) {
      oriented_bars->emplace_back();
      oriented_bars->back().emplace_back(bar_ccs[i]);
      mean_vectors->emplace_back(refl * get<0>(bar_orientations[i]),
                                 refl * get<1>(bar_orientations[i]));
    } else {
      (*oriented_bars)[insert_idx].emplace_back(bar_ccs[i]);
      int n = (*oriented_bars)[insert_idx].size();
      double mean_x = get<0>((*mean_vectors)[insert_idx]),
             mean_y = get<1>((*mean_vectors)[insert_idx]);
      mean_x = ((n - 1) * mean_x + refl * get<0>(bar_orientations[i])) / n;
      mean_y = ((n - 1) * mean_y + refl * get<1>(bar_orientations[i])) / n;
      double mag = sqrt(mean_x * mean_x + mean_y * mean_y);
      get<0>((*mean_vectors)[insert_idx]) = mean_x / mag;
      get<1>((*mean_vectors)[insert_idx]) = mean_y / mag;
    }
  }

  for (size_t i = 0; i < 2; i++) {
    size_t max_size = 0;
    int max_idx = -1;
    for (size_t j = i; j < oriented_bars->size(); j++) {
      if ((*oriented_bars)[j].size() > max_size) {
        max_size = (*oriented_bars)[j].size();
        max_idx = j;
      }
    }
    if (max_idx >= 0) {
      swap((*oriented_bars)[max_idx], (*oriented_bars)[i]);
      swap((*mean_vectors)[max_idx], (*mean_vectors)[i]);
    }
  }

  if (oriented_bars->size() > 2) {
    oriented_bars->resize(2);
    mean_vectors->resize(2);
  }
}


void Usaf1951Target::DetectTriBars(const vector<int>& oriented_bars,
                                   const Vector2d& mean_vector,
                                   const Mat_<double>& cc_centroids,
                                   vector<TriBar>* tribars) const {
  const double kMaxCentroidOffset = 5;  // [pixels]
  const double kMaxDistanceDifference = 0.05; // [percent]

  vector<bool> available(oriented_bars.size(), true);
  for (size_t i = 0; i < oriented_bars.size(); i++) {
    if (!available[i]) continue;

    vector<pair<int, double>> neighbors{make_pair(-1, 1e10),
                                        make_pair(-1, 1e10)};
    for (size_t j = 0; j < oriented_bars.size(); j++) {
      if (i == j || !available[j]) continue;

      double dx = cc_centroids(oriented_bars[j], 0) -
                  cc_centroids(oriented_bars[i], 0);
      double dy = cc_centroids(oriented_bars[j], 1) -
                  cc_centroids(oriented_bars[i], 1);

      double orth_dist = dx * get<0>(mean_vector) + dy * get<1>(mean_vector);

      if (abs(orth_dist) > kMaxCentroidOffset) continue;

      double dist = sqrt(dx*dx + dy*dy);

      if (dist < neighbors[0].second) {
        neighbors[1] = neighbors[0];
        neighbors[0] = make_pair(j, dist);
      } else if (dist < neighbors[1].second) {
        neighbors[1] = make_pair(j, dist);
      }
    }

    if (neighbors[1].first == -1) continue;

    double percent_diff = (neighbors[1].second - neighbors[0].second) /
                          neighbors[0].second;
    if (percent_diff < kMaxDistanceDifference) {
      tribars->emplace_back(oriented_bars[i],
                            oriented_bars[neighbors[0].first],
                            oriented_bars[neighbors[1].first]);
      available[i] = false;
      available[neighbors[0].first] = false;
      available[neighbors[1].first] = false;
    }
  }
}


double Usaf1951Target::BarArea(const TriBar& tribar,
                               const Mat_<int32_t>& cc_stats) const {
  return (cc_stats(get<0>(tribar), CC_STAT_AREA) +
          cc_stats(get<1>(tribar), CC_STAT_AREA) +
          cc_stats(get<2>(tribar), CC_STAT_AREA)) / 3.;
}


void Usaf1951Target::AnalyzeAreaRatios(const vector<vector<TriBar>>& bar_groups,
                                       const Mat_<int32_t>& cc_stats,
                                       vector<vector<double>>* area_ratios,
                                       double* median_ratio,
                                       double* ratio_spread) const {
  vector<double> area_ratio_mdn_vec;
  *ratio_spread = 0;
  double ratio_mean = 0;
  int ratio_count = 0;

  // Find the mean and standard deviation of the area ratios
  for (size_t i = 0; i < bar_groups.size(); i++) {
    area_ratios->emplace_back();
    for (size_t j = 1; j < bar_groups[i].size(); j++) {
      double area_ratio = BarArea(bar_groups[i][j], cc_stats) /
                          BarArea(bar_groups[i][j-1], cc_stats);
      area_ratios->back().emplace_back(area_ratio);
      area_ratio_mdn_vec.push_back(area_ratio);
      *ratio_spread += pow(area_ratio, 2);
      ratio_mean += area_ratio;
      ratio_count++;
    }
  }

  // Calculate the standard deviation
  *ratio_spread = sqrt(*ratio_spread / ratio_count -
                       pow(ratio_mean / ratio_count, 2));

  // Find the median ratio
  size_t median_n = area_ratio_mdn_vec.size() / 2;
  nth_element(area_ratio_mdn_vec.begin(), area_ratio_mdn_vec.begin() + median_n,
              area_ratio_mdn_vec.end());
  *median_ratio = area_ratio_mdn_vec[median_n];
}


void Usaf1951Target::DetectMisses(vector<vector<TriBar>>& bar_groups,
                                  const Mat_<int32_t>& cc_stats) const {
  // Analyze the area ratios between groups.
  vector<vector<double>> area_ratios;
  double area_ratio_exp, area_ratio_spread;
  AnalyzeAreaRatios(bar_groups, cc_stats,
                    &area_ratios, &area_ratio_exp, &area_ratio_spread);
  
  // It's possible we found all of them (wouldn't that be nice!), so set a
  // minimum value for the spread.
  area_ratio_spread = max(area_ratio_spread, 0.05);


  const size_t kExpectedTriBars = num_levels_ * 12;
  while (bar_groups[0].size() < kExpectedTriBars ||
         bar_groups[1].size() < kExpectedTriBars) {
    vector<double> first_areas;
    for (size_t i = 0; i < bar_groups.size(); i++) {
      double area = 1;
      int index = 0;
      while (get<0>(bar_groups[i][index]) < 0) {
        index++;
        area /= area_ratio_exp;
      }
      area *= BarArea(bar_groups[i][index], cc_stats);
      first_areas.push_back(area);
    }

    double first_ratio = min(first_areas[0], first_areas[1]) /
                         max(first_areas[0], first_areas[1]);
    
    // If the ratio between the first two elements is closer to the expected
    // ratio for subsequent groups than 1, we can assume that one of the two
    // orientations is missing the first group, so insert a blank one.
    if ((first_ratio - area_ratio_exp) < (1 - first_ratio)) {
      if (first_areas[0] > first_areas[1]) {
        bar_groups[1].emplace(begin(bar_groups[1]), -1, -1, -1);
        area_ratios[1].emplace(begin(area_ratios[1]), area_ratio_exp);
      } else {
        bar_groups[0].emplace(begin(bar_groups[0]), -1, -1, -1);
        area_ratios[0].emplace(begin(area_ratios[0]), area_ratio_exp);
      }
    } else break;
  }

  bool found_miss = true;
  while (found_miss) {
    found_miss = false;
    for (size_t i = 0; i < bar_groups.size(); i++) {
      for (size_t j = 1; j < bar_groups[i].size(); j++) {
        double z_score = abs(area_ratios[i][j-1] - area_ratio_exp) /
                         area_ratio_spread;
        if (z_score > 2) {
          bar_groups[i].emplace(begin(bar_groups[i]) + j, -1, -1, -1);
          area_ratios[i].emplace(begin(area_ratios[i]) + j - 1, area_ratio_exp);
          area_ratios[i][j] /= area_ratio_exp;
          found_miss = true;
          break;
        }
      }
    }
  }

  for (size_t i = 0; i < bar_groups.size(); i++) {
    while (bar_groups[i].size() < kExpectedTriBars) {
      bar_groups[i].emplace_back(-1, -1, -1);
    }
  }
}


void Usaf1951Target::FindBoundingBoxes(
    const vector<vector<TriBar>>& bar_groups,
    const Mat_<int32_t>& cc_labels,
    const Mat_<int32_t>& cc_stats,
    const vector<Vector2d>& mean_vectors,
    vector<vector<BoundingBox>>* bounding_boxes,
    vector<vector<Vector2d>>* bb_centroids) const {
  Matx<double, 2, 2> rot(get<0>(mean_vectors[0]), get<1>(mean_vectors[0]),
                         get<0>(mean_vectors[1]), get<1>(mean_vectors[1]));
  Matx<double, 2, 2> rot_inv = rot.inv();

  for (size_t i = 0; i < bar_groups.size(); i++) {
    bounding_boxes->emplace_back();
    bb_centroids->emplace_back();
    for (size_t j = 0; j < bar_groups[i].size(); j++) {
      bounding_boxes->back().emplace_back();
      BoundingBox& box = bounding_boxes->back().back();

      int bar1, bar2, bar3;
      std::tie(bar1, bar2, bar3) = bar_groups[i][j];

      double min_x = 0, min_y = 0, max_x = 0, max_y = 0;
      if (bar1 == -1) {
        box.resize(8, 0);
        bb_centroids->back().emplace_back(-1, -1);
        continue;
      }

      min_x = min({cc_stats(bar1, CC_STAT_LEFT), cc_stats(bar2, CC_STAT_LEFT),
                   cc_stats(bar3, CC_STAT_LEFT)});
      min_y = min({cc_stats(bar1, CC_STAT_TOP), cc_stats(bar2, CC_STAT_TOP),
                   cc_stats(bar3, CC_STAT_TOP)});
      max_x = max({
          cc_stats(bar1, CC_STAT_LEFT) + cc_stats(bar1, CC_STAT_WIDTH),
          cc_stats(bar2, CC_STAT_LEFT) + cc_stats(bar2, CC_STAT_WIDTH),
          cc_stats(bar3, CC_STAT_LEFT) + cc_stats(bar3, CC_STAT_WIDTH)});
      max_y = max({
          cc_stats(bar1, CC_STAT_TOP) + cc_stats(bar1, CC_STAT_HEIGHT),
          cc_stats(bar2, CC_STAT_TOP) + cc_stats(bar2, CC_STAT_HEIGHT), 
          cc_stats(bar3, CC_STAT_TOP) + cc_stats(bar3, CC_STAT_HEIGHT)});


      double min_x2 = 1e10, min_y2 = 1e10, max_x2 = -1e10, max_y2 = -1e10;
      for (int y = min_y; y < max_y; y++) {
        for (int x = min_x; x < max_x; x++) {
          int label = cc_labels(y, x);
          if (label != bar1 && label != bar2 && label != bar3) continue;

          double x2 = rot(0, 0) * x + rot(0, 1) * y;
          double y2 = rot(1, 0) * x + rot(1, 1) * y;

          min_x2 = min(min_x2, x2);
          max_x2 = max(max_x2, x2);
          min_y2 = min(min_y2, y2);
          max_y2 = max(max_y2, y2);
        }
      }

      box.push_back(rot_inv(0, 0) * min_x2 + rot_inv(0, 1) * min_y2);
      box.push_back(rot_inv(1, 0) * min_x2 + rot_inv(1, 1) * min_y2);
      box.push_back(rot_inv(0, 0) * min_x2 + rot_inv(0, 1) * max_y2);
      box.push_back(rot_inv(1, 0) * min_x2 + rot_inv(1, 1) * max_y2);
      box.push_back(rot_inv(0, 0) * max_x2 + rot_inv(0, 1) * max_y2);
      box.push_back(rot_inv(1, 0) * max_x2 + rot_inv(1, 1) * max_y2);
      box.push_back(rot_inv(0, 0) * max_x2 + rot_inv(0, 1) * min_y2);
      box.push_back(rot_inv(1, 0) * max_x2 + rot_inv(1, 1) * min_y2);

      bb_centroids->back().emplace_back(
          0.25 * (box[0] + box[2] + box[4] + box[6]),
          0.25 * (box[1] + box[3] + box[5] + box[7]));
    }
  }
}


void Usaf1951Target::CompletePartialPairs(
    vector<vector<BoundingBox>>& bounding_boxes,
    vector<vector<Vector2d>>& bb_centroids) const {
  // Fill in the offsets between the complete pairs. Keep track of the mean
  // vector (make sure it has a +x, so they don't all cancel out).
  vector<pair<int, double>> distances;
  double mean_dx = 0, mean_dy = 0;
  for (size_t i = 0; i < bb_centroids[0].size(); i++) {
    if (get<0>(bb_centroids[0][i]) == -1 || get<0>(bb_centroids[1][i]) == -1) {
      continue;
    }

    double dx = get<0>(bb_centroids[0][i]) - get<0>(bb_centroids[1][i]);
    double dy = get<1>(bb_centroids[0][i]) - get<1>(bb_centroids[1][i]);

    double dist = sqrt(dx * dx + dy * dy);
    distances.emplace_back(i, dist);

    if (dx < 0) {
      dx *= -1; dy *= -1;
    }
    mean_dx += dx;
    mean_dy += dy;
  }
  double mean_mag = sqrt(mean_dx * mean_dx + mean_dy * mean_dy);
  mean_dx /= mean_mag;
  mean_dy /= mean_mag;

  // The distance between the centers of the pairs of tribars decays by a factor
  // of ~0.89 every pair, so we'll use an exponential model.
  // 
  // distance = initial_value * (mean_ratio)^pair_index
  double mean_ratio = 0;
  int count = 0;
  for (size_t i = 1; i < distances.size(); i++) {
    if ((distances[i].first - distances[i-1].first) != 1) continue;

    double ratio = distances[i].second / distances[i-1].second;
    mean_ratio = (count * mean_ratio + ratio) / (count + 1);
    count++;
  }

  double initial_value = distances[0].second / pow(mean_ratio,
                                                   distances[0].first);

  // In order to fill in the pairs where we only have one of the bars, we'll
  // need to know the orientation of the mean offset vector. While we don't know
  // the orientation of the image, the standard tells us when the orientaation
  // switches, so we can fit that knowledge to the data.
  vector<int> offset_orient{-1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1};
  vector<double> dot_products;
  for (size_t i = 0; i < offset_orient.size(); i++) {
    if (get<0>(bb_centroids[0][i]) == -1 || get<0>(bb_centroids[1][i]) == -1) {
      dot_products.push_back(0);
    } else {
      dot_products.push_back(
          mean_dx * (get<0>(bb_centroids[1][i]) - get<0>(bb_centroids[0][i])) +
          mean_dy * (get<0>(bb_centroids[1][i]) - get<0>(bb_centroids[0][i])));
    }
  }

  int num_pos = 0, num_neg = 0;
  for (size_t i = 0; i < offset_orient.size(); i++) {
    if (dot_products[i] == 0) continue;

    double dot = dot_products[i] * offset_orient[i];
    if (dot > 0) num_pos++;
    if (dot < 0) num_neg++;
  }

  if (num_pos > 0 && num_neg > 0) {
    cerr << "Error: Relative orientation of horizontal/vertical bars "
         << "does not match up with the standard USAF-1951 pattern."
         << endl;
    return;
  }

  if (num_neg > 0) for (auto& tmp : offset_orient) tmp *= -1;

  for (size_t i = 0; i < bb_centroids[0].size(); i++) {
    double pred_dist = initial_value * pow(mean_ratio, i);
    int dest = -1;

    if (get<0>(bb_centroids[0][i]) == -1 && get<0>(bb_centroids[1][i]) != -1) {
      dest = 0;
    } else if (get<0>(bb_centroids[0][i]) != -1 &&
               get<0>(bb_centroids[1][i]) == -1) {
      dest = 1;
    } else continue;

    int refl = dest == 0 ? -1 : 1;
    double dx = refl * pred_dist * mean_dx * offset_orient[i % 12];
    double dy = refl * pred_dist * mean_dy * offset_orient[i % 12];

    get<0>(bb_centroids[dest][i]) = get<0>(bb_centroids[1-dest][i]) + dx;
    get<1>(bb_centroids[dest][i]) = get<1>(bb_centroids[1-dest][i]) + dy;
    for (size_t j = 0; j < bounding_boxes[1-dest][i].size(); j += 2) {
      bounding_boxes[dest][i][j] = bounding_boxes[1-dest][i][j] + dx;
      bounding_boxes[dest][i][j+1] = bounding_boxes[1-dest][i][j+1] + dy;
    }
  }
}


void Usaf1951Target::CompleteLowerLevel(
    vector<vector<BoundingBox>>& bounding_boxes,
    vector<vector<Vector2d>>& bb_centroids,
    int level) const {
  const int kGroupsPerLevel = 12;

  const size_t kLowerStart = level * kGroupsPerLevel;
  const size_t kLowerEnd = (level + 1) * kGroupsPerLevel;

  if (kLowerEnd > bb_centroids[0].size()) {
    cerr << "Error: the given level was beyond the number of present levels."
         << endl;
    return;
  }

  // In order to fit the transform, we need corresponding points between the two
  // levels. We'll use the corners of bounding boxes that were found in both
  // levels. We need 3 to find the parameters.
  int num_correspondences = 0;
  vector<vector<int>> correspondences;
  for (const auto& centroids : bb_centroids) {
    correspondences.emplace_back();
    for (size_t i = kLowerStart; i < kLowerEnd; i++) {
      if (get<0>(centroids[i]) != -1 &&
          get<0>(centroids[i-kLowerStart]) != -1) {
        num_correspondences += 4;
        correspondences.back().emplace_back(i);
      }
    }
  }

  if (num_correspondences < 3) {
    cerr << "Error: not enough correspondences found with the requested level."
         << endl;
    return;
  }

  // Perform a least squares to find the transform.
  //
  // [x_1 1 0]        [x'_1]
  // [y_1 0 1]        [y'_1]
  // [.......] [ s] = [....]
  // [x_n 1 0] [dx]   [x'_n]
  // [y_n 0 1] [dy]   [y'_n]
  int current_row = 0;
  Mat_<double> obs_matrix(2 * num_correspondences, 3);
  Mat_<double> result_vector(2 * num_correspondences, 1);
  for (size_t i = 0; i < correspondences.size(); i++) {
    for (auto cor : correspondences[i]) {
      for (int j = 0; j < 8; j++) {
        obs_matrix(current_row, 0) = bounding_boxes[i][cor - kLowerStart][j];
        obs_matrix(current_row, 1) = (j % 2) == 0 ? 1 : 0;
        obs_matrix(current_row, 2) = 1 - obs_matrix(current_row, 1);
        result_vector(current_row, 0) = bounding_boxes[i][cor][j];
        current_row++;
      }
    }
  }

  Mat_<double> params = (obs_matrix.t() * obs_matrix).inv() *
                        obs_matrix.t() * result_vector;

  for (size_t i = 0; i < bounding_boxes.size(); i++) {
    for (size_t j = kLowerStart; j < kLowerEnd; j++) {
      if (get<0>(bb_centroids[i][j]) != -1) continue;
      if (get<0>(bb_centroids[i][j-kLowerStart]) == -1) continue;

      double centroid_x = 0;
      double centroid_y = 0;
      for (size_t k = 0; k < bounding_boxes[i][j].size(); k += 2) {
        double pred_x = params(0) * bounding_boxes[i][j-kLowerStart][k] +
                        params(1);
        double pred_y = params(0) * bounding_boxes[i][j-kLowerStart][k+1] +
                        params(2);

        bounding_boxes[i][j][k] = pred_x;
        bounding_boxes[i][j][k+1] = pred_y;
        centroid_x += pred_x / (0.5 * bounding_boxes[i][j].size());
        centroid_y += pred_y / (0.5 * bounding_boxes[i][j].size());
      }
      get<0>(bb_centroids[i][j]) = centroid_x;
      get<1>(bb_centroids[i][j]) = centroid_y;
    }
  }
}
