// File Description
// Author: Philip Salvaggio

#include "usaf_1951_target.h"

#include "base/assertions.h"
#include "base/opencv_utils.h"
#include "otf_measurement/slant_edge_mtf.h"

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
  mean_vectors_.clear();
  vector<vector<int>> oriented_bars;
  SplitHorizontalVerticalBars(bar_ccs, bar_orientations,
                              &oriented_bars, &mean_vectors_);

  if (oriented_bars.size() < 2) {
    cerr << "Error: Only one orientation of bars found." << endl;
    return false;
  }

  // For each orientation, detect groups of tri-bars and sort them in decreasing
  // bar size.
  vector<vector<TriBar>> bar_groups;
  for (size_t i = 0; i < oriented_bars.size(); i++) {
    bar_groups.emplace_back();
    DetectTriBars(oriented_bars[i], mean_vectors_[i], cc_centroids,
                  &(bar_groups.back()));

    sort(begin(bar_groups.back()), end(bar_groups.back()),
        [&cc_stats] (const TriBar& a, const TriBar& b) {
          return cc_stats(get<0>(a), CC_STAT_AREA) >
                 cc_stats(get<0>(b), CC_STAT_AREA);
        });
  }

  // Detect the missed tri-bar groups. This modifies bar_groups with blanks in
  // the place of the missed groups.
  DetectMisses(bar_groups, cc_stats);

  // Find the bounding boxes around each of the tri-bar groups. These bounding
  // boxes are not axis-aligned, but oriented with the pattern.
  vector<vector<Vector2d>> bb_centroids;
  FindBoundingBoxes(bar_groups, cc_labels, cc_stats, mean_vectors_,
                    &bounding_boxes_);

  // Infer the locations of the tri-bar groups where the horizontal or the
  // vertical group was found, but not both.
  CompletePartialPairs(bounding_boxes_);

  // Infer the location of missing tri-bars in lower levels by using the
  // locations in the biggest level.
  for (int i = 1; i < num_levels_; i++) {
    CompleteLowerLevel(bounding_boxes_, i);
  }

  // Make sure the horiontal tri-bars are first in bounding_boxes_ and
  // mean_vectors_.
  if (!IsHorizontalFirst(bounding_boxes_, mean_vectors_)) {
    double tmp;
    for (int i = 0; i < bounding_boxes_.rows; i += 2) {
      for (int j = 0; j < bounding_boxes_.cols; j++) {
        tmp = bounding_boxes_(i, j);
        bounding_boxes_(i, j) = bounding_boxes_(i+1, j);
        bounding_boxes_(i+1, j) = tmp;
      }
    }
    swap(mean_vectors_[0], mean_vectors_[1]);
  }

  return true;
}


bool Usaf1951Target::FoundBarGroup(int bar_group) const {
  return bounding_boxes_(bar_group, 8) != -1;
}


void Usaf1951Target::GetProfile(int bar_group,
                                int orientation,
                                vector<pair<double, double>>* profile) const {
  CHECK(profile);
  CHECK(orientation == HORIZONTAL || orientation == VERTICAL);
  CHECK(FoundBarGroup(bar_group));

  profile->clear();

  BoundingBox box;
  GetProfileRegion(bar_group, orientation, &box);
  
  // Get profile direction unit vector
  double prof_x = 0, prof_y = 0;
  double side1_len2 = pow(box[2] - box[0], 2) + pow(box[3] - box[1], 2),
         side2_len2 = pow(box[4] - box[2], 2) + pow(box[5] - box[3], 2);
  if (side1_len2 > side2_len2) {
    double mag = sqrt(side1_len2);
    prof_x = (box[2] - box[0]) / mag;
    prof_y = (box[3] - box[1]) / mag;
  } else {
    double mag = sqrt(side2_len2);
    prof_x = (box[4] - box[2]) / mag;
    prof_y = (box[5] - box[3]) / mag;
  }

  // Get profile endpoints
  double endpoint1_x, endpoint1_y, endpoint2_x, endpoint2_y;
  if (side1_len2 > side2_len2) {
    endpoint1_x = 0.5 * (box[4] + box[2]);
    endpoint1_y = 0.5 * (box[5] + box[3]);
    endpoint2_x = 0.5 * (box[6] + box[0]);
    endpoint2_y = 0.5 * (box[7] + box[1]);
  } else {
    endpoint1_x = 0.5 * (box[0] + box[2]);
    endpoint1_y = 0.5 * (box[1] + box[3]);
    endpoint2_x = 0.5 * (box[4] + box[6]);
    endpoint2_y = 0.5 * (box[5] + box[7]);
  }

  // Determine the start and end of the profile
  double prof_start_x, prof_start_y, prof_end_x, prof_end_y;
  double dot1 = endpoint1_x * prof_x + endpoint1_y * prof_y,
         dot2 = endpoint2_x * prof_x + endpoint2_y * prof_y;
  double start_dot = min(dot1, dot2);
  if (dot1 < dot2) {
    prof_start_x = endpoint1_x;
    prof_start_y = endpoint1_y;
    prof_end_x = endpoint2_x;
    prof_end_y = endpoint2_y;
  } else {
    prof_start_x = endpoint2_x;
    prof_start_y = endpoint2_y;
    prof_end_x = endpoint1_x;
    prof_end_y = endpoint1_y;
  }

  // Determine image bounds round profile region
  int min_row = numeric_limits<int>::max(),
      min_col = numeric_limits<int>::max(),
      max_row = -1,
      max_col = -1;
  for (size_t i = 0; i < box.size(); i += 2) {
    min_col = min(int(floor(box[i])), min_col);
    max_col = max(int(ceil(box[i])), max_col);
    min_row = min(int(floor(box[i+1])), min_row);
    max_row = max(int(ceil(box[i+1])), max_row);
  }

  // Create the initial profile.
  Mat_<uint8_t> mask(max_row - min_row + 1, max_col - min_col + 1);
  for (int i = min_row; i <= max_row; i++) {
    for (int j = min_col; j <= max_col; j++) {
      mask(i - min_row, j - min_col) = PointInQuad(j, i, box) ? 1 : 0;
    }
  }

  Mat roi = image_(Range(min_row, max_row + 1), Range(min_col, max_col + 1));
  
  double edge[3];
  edge[0] = prof_x;
  edge[1] = prof_y;
  edge[2] = prof_x * 0.5 * (prof_start_x + prof_end_x) +
            prof_y * 0.5 * (prof_start_y + prof_end_y) - start_dot;

  SlantEdgeMtf slant_edge;
  int samples = slant_edge.GetSamplesPerPixel(roi, edge);

  vector<double> prof, prof_stddevs;
  slant_edge.GenerateEsf(roi, edge, samples, &prof, &prof_stddevs, mask);
  slant_edge.SmoothEsf(&prof);

  for (size_t i = 0; i < prof.size(); i++) {
    profile->emplace_back(double(i) / samples, prof[i]);
  }
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
  for (int i = 0; i < bounding_boxes_.rows; i++) {
    if (bounding_boxes_(i, 8) != -1) {
      for (size_t j = 0; j < 8; j += 2) {
        line(bars,
             Point(bounding_boxes_(i, j), bounding_boxes_(i, j+1)),
             Point(bounding_boxes_(i, (j+2) % 8),
                   bounding_boxes_(i, (j+3) % 8)),
               255 * colors[(i / 2) % colors.size()]);
      }
    }
  }

  return bars;
}


Mat Usaf1951Target::VisualizeProfileRegions() const {
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
  for (int i = 0; i < bounding_boxes_.rows; i += 2) {
    for (size_t j = 0; j < 2; j++) {
      BoundingBox region;
      GetProfileRegion(i / 2, j, &region);

      for (size_t k = 0; k < 8; k += 2) {
        line(bars,
             Point(region[k], region[k+1]),
             Point(region[(k+2) % 8], region[(k+3) % 8]),
             255 * colors[(i / 2) % colors.size()]);
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


  const size_t kExpectedTriBars = num_levels_ * kTriBarPairsPerLevel;
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
        if (z_score > 1.5) {
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
    Mat_<double>* bounding_boxes) const {

  Mat_<double>& boxes = *bounding_boxes;
  boxes.create(num_levels_ * kTriBarsPerLevel, 10);
  boxes = 0;

  Matx<double, 2, 2> rot(get<0>(mean_vectors[0]), get<1>(mean_vectors[0]),
                         get<0>(mean_vectors[1]), get<1>(mean_vectors[1]));
  Matx<double, 2, 2> rot_inv = rot.inv();

  for (size_t i = 0; i < bar_groups.size(); i++) {
    for (size_t j = 0; j < bar_groups[i].size(); j++) {
      int idx = 2 * j + i;

      int bar1, bar2, bar3;
      std::tie(bar1, bar2, bar3) = bar_groups[i][j];

      double min_x = 0, min_y = 0, max_x = 0, max_y = 0;
      if (bar1 == -1) {
        boxes(idx, 8) = -1;
        boxes(idx, 9) = -1;
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

      boxes(idx, 0) = rot_inv(0, 0) * min_x2 + rot_inv(0, 1) * min_y2;
      boxes(idx, 1) = rot_inv(1, 0) * min_x2 + rot_inv(1, 1) * min_y2;
      boxes(idx, 2) = rot_inv(0, 0) * min_x2 + rot_inv(0, 1) * max_y2;
      boxes(idx, 3) = rot_inv(1, 0) * min_x2 + rot_inv(1, 1) * max_y2;
      boxes(idx, 4) = rot_inv(0, 0) * max_x2 + rot_inv(0, 1) * max_y2;
      boxes(idx, 5) = rot_inv(1, 0) * max_x2 + rot_inv(1, 1) * max_y2;
      boxes(idx, 6) = rot_inv(0, 0) * max_x2 + rot_inv(0, 1) * min_y2;
      boxes(idx, 7) = rot_inv(1, 0) * max_x2 + rot_inv(1, 1) * min_y2;
      boxes(idx, 8) = 0.25 * (boxes(idx, 0) + boxes(idx, 2) + boxes(idx, 4) +
                              boxes(idx, 6));
      boxes(idx, 9) = 0.25 * (boxes(idx, 1) + boxes(idx, 3) + boxes(idx, 5) +
                              boxes(idx, 7));
    }
  }
}


void Usaf1951Target::CompletePartialPairs(Mat_<double>& bounding_boxes) const {
  // Fill in the offsets between the complete pairs. Keep track of the mean
  // vector (make sure it has a +x, so they don't all cancel out).
  vector<pair<int, double>> distances;
  double mean_dx = 0, mean_dy = 0;
  for (int i = 0; i < bounding_boxes.rows; i += 2) {
    if (bounding_boxes(i, 8) == -1 || bounding_boxes(i+1, 8) == -1) {
      continue;
    }

    double dx = bounding_boxes(i, 8) - bounding_boxes(i+1, 8);
    double dy = bounding_boxes(i, 9) - bounding_boxes(i+1, 9);

    double dist = sqrt(dx * dx + dy * dy);
    distances.emplace_back(i/2, dist);

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
    if (bounding_boxes(2*i, 8) == -1 || bounding_boxes(2*i+1, 8) == -1) {
      dot_products.push_back(0);
    } else {
      dot_products.push_back(
          mean_dx * (bounding_boxes(2*i+1, 8) - bounding_boxes(2*i, 8)) +
          mean_dy * (bounding_boxes(2*i+1, 9) - bounding_boxes(2*i, 9)));
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

  for (int i = 0; i < bounding_boxes.rows; i += 2) {
    double pred_dist = initial_value * pow(mean_ratio, i / 2);
    int dest = -1, src = -1;

    if (bounding_boxes(i, 8) == -1 && bounding_boxes(i+1, 8) != -1) {
      dest = i; src = i + 1;
    } else if (bounding_boxes(i, 8) != -1 && bounding_boxes(i+1, 8) == -1) {
      dest = i + 1; src = i;
    } else continue;

    int refl = dest == i ? -1 : 1;
    double dx = refl * pred_dist * mean_dx *
                offset_orient[(i/2) % kTriBarPairsPerLevel];
    double dy = refl * pred_dist * mean_dy *
                offset_orient[(i/2) % kTriBarPairsPerLevel];

    for (int j = 0; j < bounding_boxes.cols; j += 2) {
      bounding_boxes(dest, j) = bounding_boxes(src, j) + dx;
      bounding_boxes(dest, j+1) = bounding_boxes(src, j+1) + dy;
    }
  }
}


void Usaf1951Target::CompleteLowerLevel(
    Mat_<double>& bounding_boxes,
    int level) const {
  const int kLowerStart = level * kTriBarsPerLevel;
  const int kLowerEnd = (level + 1) * kTriBarsPerLevel;

  if (kLowerEnd > bounding_boxes.rows) {
    cerr << "Error: the given level was beyond the number of present levels."
         << endl;
    return;
  }

  // In order to fit the transform, we need corresponding points between the two
  // levels. We'll use the corners of bounding boxes that were found in both
  // levels. We need 3 to find the parameters.
  int num_correspondences = 0;
  vector<int> correspondences;
  for (int i = 0; i < kTriBarsPerLevel; i++) {
    if (bounding_boxes(i, 8) != -1 && bounding_boxes(i+kLowerStart, 8) != -1) {
      num_correspondences += 4;
      correspondences.emplace_back(i);
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
  for (auto cor : correspondences) {
    for (int i = 0; i < 8; i++) {
      obs_matrix(current_row, 0) = bounding_boxes(cor, i);
      obs_matrix(current_row, 1) = (i % 2) == 0 ? 1 : 0;
      obs_matrix(current_row, 2) = 1 - obs_matrix(current_row, 1);
      result_vector(current_row, 0) = bounding_boxes(cor+kLowerStart, i);
      current_row++;
    }
  }

  Mat_<double> params = (obs_matrix.t() * obs_matrix).inv() *
                        obs_matrix.t() * result_vector;

  for (int i = kLowerStart; i < kLowerEnd; i++) {
    if (bounding_boxes(i, 8) != -1) continue;
    if (bounding_boxes(i-kLowerStart, 8) == -1) continue;

    double centroid_x = 0;
    double centroid_y = 0;
    for (size_t k = 0; k < 8; k += 2) {
      double pred_x = params(0) * bounding_boxes(i-kLowerStart, k) +
                      params(1);
      double pred_y = params(0) * bounding_boxes(i-kLowerStart, k+1) +
                      params(2);

      bounding_boxes(i, k) = pred_x;
      bounding_boxes(i, k+1) = pred_y;
      centroid_x += 0.25 * pred_x;
      centroid_y += 0.25 * pred_y;
    }
    bounding_boxes(i, 8) = centroid_x;
    bounding_boxes(i, 9) = centroid_y;
  }
}

bool Usaf1951Target::IsHorizontalFirst(
       const Mat_<double>& bounding_boxes,
       const vector<Vector2d>& mean_vectors) const {
  CHECK(bounding_boxes.rows >= kTriBarsPerLevel);

  // The key feature here is that horizontal bars are always on the outside of
  // the target. We'll confine the analysis to the first group. We'll find the
  // vector that goes in the x-direction on the target (between horizontal and
  // vertical groups) and find the tri-bar group that has the maximum total
  // absolute dot product with that vector.
  double dx = bounding_boxes(0, 8) - bounding_boxes(1, 8);
  double dy = bounding_boxes(0, 9) - bounding_boxes(1, 9);

  Vector2d target_orientation;
  if (abs(dx * get<0>(mean_vectors[0]) + dy * get<1>(mean_vectors[0])) >
      abs(dx * get<0>(mean_vectors[1]) + dy * get<1>(mean_vectors[1]))) {
    target_orientation = mean_vectors[0];
  } else {
    target_orientation = mean_vectors[1];
  }

  double mean[2];
  mean[0] = 0; mean[1] = 0;
  for (int i = 0; i < kTriBarsPerLevel; i++) {
    mean[0] += bounding_boxes(i, 8);
    mean[1] += bounding_boxes(i, 9);
  }
  mean[0] /= kTriBarsPerLevel; mean[1] /= kTriBarsPerLevel;

  double dots[2];
  for (int i = 0; i < 2; i++) {
    dots[i] = 0;
    for (int j = 0; j < kTriBarPairsPerLevel; j++) {
      dots[i] += abs(
        get<0>(target_orientation) * (bounding_boxes(2*j + i, 8) - mean[0]) +
        get<1>(target_orientation) * (bounding_boxes(2*j + i, 9) - mean[1]));
    }
  }

  return dots[0] > dots[1];
}


void Usaf1951Target::GetProfileRegion(int bar_group,
                                      int orientation,
                                      BoundingBox* region) const {
  CHECK(bar_group < bounding_boxes_.rows / 2,
        "Error: must call RecognizeTarget() before getting profiles.");

  Mat_<double> box = bounding_boxes_.row(2 * bar_group + orientation);
  double profile_dir_x = get<0>(mean_vectors_[1-orientation]);
  double profile_dir_y = get<1>(mean_vectors_[1-orientation]);
  double profile_orth_dir_x = -profile_dir_y;
  double profile_orth_dir_y = profile_dir_x;

  double side_length = 0;
  double centroid_x = 0, centroid_y = 0;
  for (int i = 0; i < 8; i += 2) {
    double dx = box(i) - box((i+2) % 8),
           dy = box(i+1) - box((i+3) % 8);
    side_length += 0.25 * sqrt(dx*dx + dy*dy);
    centroid_x += 0.25 * box(i);
    centroid_y += 0.25 * box(i+1);
  }

  double profile_length = side_length * 7. / 5.;
  double profile_width = side_length / 3.;

  region->push_back(centroid_x - 0.5 * profile_length * profile_dir_x
                               - 0.5 * profile_width * profile_orth_dir_x);
  region->push_back(centroid_y - 0.5 * profile_length * profile_dir_y
                               - 0.5 * profile_width * profile_orth_dir_y);

  region->push_back(centroid_x - 0.5 * profile_length * profile_dir_x
                               + 0.5 * profile_width * profile_orth_dir_x);
  region->push_back(centroid_y - 0.5 * profile_length * profile_dir_y
                               + 0.5 * profile_width * profile_orth_dir_y);

  region->push_back(centroid_x + 0.5 * profile_length * profile_dir_x
                               + 0.5 * profile_width * profile_orth_dir_x);
  region->push_back(centroid_y + 0.5 * profile_length * profile_dir_y
                               + 0.5 * profile_width * profile_orth_dir_y);

  region->push_back(centroid_x + 0.5 * profile_length * profile_dir_x
                               - 0.5 * profile_width * profile_orth_dir_x);
  region->push_back(centroid_y + 0.5 * profile_length * profile_dir_y
                               - 0.5 * profile_width * profile_orth_dir_y);
}


bool Usaf1951Target::PointInQuad(double x,
                                 double y,
                                 const BoundingBox& quad) const {

  double x0 = x - quad[0], y0 = y - quad[1];

  // We'll split the quad into two triangles and see if the point is in either
  // one. The triangles are vertices 123 and 134.
  for (int i = 2; i < 7; i += 4) {
    double tri1_v0_x = quad[4] - quad[0],
           tri1_v0_y = quad[5] - quad[1],
           tri1_v1_x = quad[i] - quad[0],
           tri1_v1_y = quad[i+1] - quad[1];

  double dot00 = tri1_v0_x*tri1_v0_x + tri1_v0_y*tri1_v0_y,
         dot01 = tri1_v0_x*tri1_v1_x + tri1_v0_y*tri1_v1_y,
         dot0p = tri1_v0_x*x0 + tri1_v0_y*y0,
         dot11 = tri1_v1_x*tri1_v1_x + tri1_v1_y*tri1_v1_y,
         dot1p = tri1_v1_x*x0 + tri1_v1_y*y0;

    double invDenom = 1 / (dot00 * dot11 - dot01 * dot01);
    double u = (dot11 * dot0p - dot01 * dot1p) * invDenom;
    double v = (dot00 * dot1p - dot01 * dot0p) * invDenom;

    if ((u >= 0) && (v >= 0) && (u + v < 1)) {
      return true;
    }
  }
  
  return false;
}
