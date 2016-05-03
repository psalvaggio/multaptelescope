// File Description
// Author: Philip Salvaggio

#ifndef USAF_1951_TARGET_H
#define USAF_1951_TARGET_H

#include "otf_measurement/usaf_1951_template.pb.h"

#include <opencv2/core.hpp>
#include <array>
#include <tuple>
#include <vector>

class Usaf1951Target {
 public:
  // Constants for the orientation in GetProfile().
  enum Orientation {
    HORIZONTAL = 0,
    VERTICAL = 1
  };

  // Standed number of tri bars in a level of the target.
  static const int kTriBarsPerLevel = 24;
  static const int kTriBarPairsPerLevel = 12;
  //static const double kAreaRatio = 0.796;

 public:
  // Constructor
  //
  // Arguments:
  //  image       An image of a USAF-1951 target. This can have a bit of
  //              background in it, but should be clipped as close as possible.
  //  num_levels  The number of target levels that are nested within each other.
  Usaf1951Target(const cv::Mat& image, int num_levels);

  // Returns the number of expected bar groups.
  int num_bar_groups() const { return num_levels_ * kTriBarPairsPerLevel; }

  // Perform recognition on the target so that profiles of the tri-bars may be
  // taken.
  //
  // Returns:
  //  Whether recognition wa successful.
  bool RecognizeTarget();

  // Use a pre-loaded template instead of recognizing the target.
  //
  // Arguments:
  //  tmpl   The template to use
  void UseTemplate(const mats::Usaf1951Template& tmpl);

  // Returns whether or not a given bar group was found.
  //
  // Arguments:
  //  bar_group  The tri-bar group of interest. Groups are sorted in decreasing
  //             order of size and there are kTriBarsPairsPerLevel groups per
  //             level of the target.
  bool FoundBarGroup(int bar_group) const;

  // Get a 1-dimenstional profile over a given tri-bar target, potentially at
  // sub-pixel resolution.
  //
  // Arguments:
  //  bar_group   The tri-bar group of interest. Groups are sorted in decreasing
  //              order of size and there are kTriBarsPairsPerLevel groups per
  //              level of the target.
  //  orientation HORIZONTAL or VERTICAL
  //  profile     Output: The profile over the requested tri-bar.
  void GetProfile(int bar_group,
                  int orientation,
                  std::vector<std::pair<double, double>>* profile) const;

  // Get a visualization of the target with the bounding boxes overlaid.
  cv::Mat VisualizeBoundingBoxes() const;

  // Get a visualization of the target with the ROIs marked that will be used
  // for taking profiles.
  cv::Mat VisualizeProfileRegions() const;

  void WriteTemplate(const std::string& filename) const;

 private:
  // Represntation of a Tri-bar group. Each element is a connected component
  // label. The first element is the middle bar, there is no guarantee about the
  // order of the second and third bar.
  using TriBar = std::array<int, 3>;

  using Vector2d = std::array<double, 2>;  // [x, y]

  using Box = std::array<double, 10>;

  // Representation of a bounding box. Since our tri-bars are not necessarily
  // axis-aligned, all for corners are given as [x0 y0 .. x3 y3]
  using BoundingBox = std::vector<double>;

  // Result for one PC in a 2D PCA analysis. The result is given as
  // [eigenvalue, eigenvector x, eigenvector y]
  using Pca2dResult = std::array<double, 3>;

 private:
  // Perform a 2D principal components analysis on a connected component (CC).
  //
  // Arguments:
  //  cc_labels     The CC label image from OpenCV's CC analysis
  //  cc_centroids  The centroids of each CC, given from OpenCV's CC analysis.
  //  label         The label that should be analyzed.
  //  results       The PCA results, will have 2 PCs.
  void PcaAnalysis(const cv::Mat_<int32_t>& cc_labels,
                   const cv::Mat_<double>& cc_centroids,
                   int label,
                   std::vector<Pca2dResult>* results) const;

  // Perform base detection of bounding boxes of tri-bars from a binary image.
  // 
  // Arguments:
  //  image_bw        Thresholded binary image (0, 1)
  //  bounding_boxes  Output: The bounding boxes for each detected tri-bar in
  //                          bar_groups. Misses will be all have 0 for the
  //                          conrers and (-1, -1) for the centroid.
  //  mean_vectors    Output: The mean orientation vector for the bars groups
  //                          in oriented_bars.
  bool DetectBoundingBoxes(const cv::Mat& image_bw,
                           std::vector<std::vector<Box>>* bounding_boxes,
                           std::vector<Vector2d>* mean_vectors) const;
                           
  // Determine which connected components (CCs) in an image are USAF target
  // bars.
  //
  // Arguments:
  //  cc_labels         The CCA label image
  //  cc_stats          Stats about the CCs. See OpenCV documentation.
  //  cc_centroids      The centroids of each CC
  //  num_ccs           The number of CCs
  //  bar_ccs           Output: List of CC labels that are bars
  //  bar_orientations  Output: Orientation vectors for each bar
  void DetermineBars(const cv::Mat_<int32_t>& cc_labels,
                     const cv::Mat_<int32_t>& cc_stats,
                     const cv::Mat_<double>& cc_centroids,
                     int num_ccs,
                     std::vector<int>* bar_ccs,
                     std::vector<Vector2d>* bar_orientations) const;

  // Given a set of CC's that are bars, cluster the bars based on orientation
  // and return the two largest groups, which should correspond to the two
  // orientions of the tri-bar groups.
  //
  // Arguments:
  //  bar_ccs           The CC labels that are bars (from DetermineBars())
  //  bar_orientations  The orientation of each bar in bar_ccs (from
  //                    DetermineBars())
  //  oriented_bars     Output: CC labels grouped by bar orientation. Guaranteed
  //                            to only be 2.
  //  mean_vectors      Output: The mean orientation vector for the bars groups
  //                            in oriented_bars.
  void SplitHorizontalVerticalBars(
      const std::vector<int>& bar_ccs,
      const cv::Mat_<int32_t>& cc_stats,
      const std::vector<Vector2d>& bar_orientations,
      std::vector<std::vector<int>>* oriented_bars,
      std::vector<Vector2d>* mean_vectors) const;

  // From a set of bars oriented in the same direction, find tri-bar groups.
  //
  // Arguments:
  //  oriented_bars  CC labels of bars in the same orientation
  //  mean_vector    The mean orientation of the bars in oriented bars
  //  cc_centroids   The CC centroid data from OpenCV's CCA.
  //  tribars        Output: The tribar groups detected.
  void DetectTriBars(const std::vector<int>& oriented_bars,
                     const Vector2d& mean_vector,
                     const cv::Mat_<int32_t>& cc_stats,
                     const cv::Mat_<double>& cc_centroids,
                     std::vector<TriBar>* tribars) const;

  // Calculate the mean area of bars in a tri-bar group.
  //
  // Arguments:
  //  tribar    The tri-bar group of interest
  //  cc_stats  The CC statistics from OpenCV's CCA
  double BarArea(const TriBar& tribar, const cv::Mat_<int32_t>& cc_stats) const;

  // Calculate the median ratio of the area of bars in subsequent bar groups.
  // If all groups have been detected, this should be a constant (around 0.8). 
  // This can be used to detect missed bar groups.
  //
  // Arguments:
  //  bar_groups    Detected tri-bar groups, grouped by orientation. These
  //                tri-bars should be sorted into order of decreasing size.
  //  cc_stats      The CC statistics from OpenCV's CCA
  //  area_ratios   Output: The bar area ratios, element i is the ratio of the
  //                        bar area in group i+1 that of group i.
  //  median_ratio  Output: The median bar area ratio. Due to misses, the mean
  //                        is biased, so this is an approximation of the mean
  //                        area ratio for non-misses.
  //  ratio_spread  Output: The spread (standard deviation) of the area ratios.
  //void AnalyzeAreaRatios(const std::vector<std::vector<TriBar>>& bar_groups,
                         //const cv::Mat_<int32_t>& cc_stats,
  void AnalyzeAreaRatios(const std::vector<std::vector<Box>>& bar_groups,
                         std::vector<std::vector<double>>* area_ratios,
                         double* median_ratio,
                         double* ratio_spread) const;

  // Detect missed detections of tri-bar groups and insert blank groups into the
  // list in their place.
  //
  // Arguments:
  //  bar_groups  Input/Output: Tri-bar groups, grouped by orientation. These
  //                            should be sorted in order of decreasing size.
  //                            This list will be modified by this routine.
  //  cc_stats    The CC statistics from OpenCV's CCA
  void DetectMisses(const std::vector<std::vector<Box>>& boxes,
                    cv::Mat_<double>& output) const;
  //void DetectMisses(std::vector<std::vector<TriBar>>& bar_groups,
  //                  const cv::Mat_<int32_t>& cc_stats) const;

  // Find the bounding boxes around each detected tri-bar group.
  //
  // Arguments:
  //  bar_groups      The tri-bar targets in the USAF-1951 target. Should be
  //                  sorted in order of decreasing sizes and have blanks for
  //                  misses (i.e. output of DetectMisses()).
  //  cc_labels       The CC label image from OpenCV's CCA.
  //  cc_stats        The CC statistics from OpenCV's CCA.
  //  mean_vectors    The mean vectors of the two tri-bar orientations from
  //                  SplitHorizontalVerticalBars().
  //  bounding_boxes  Output: The bounding boxes for each detected tri-bar in
  //                          bar_groups. Misses will be all have 0 for the
  //                          conrers and (-1, -1) for the centroid.
  void FindBoundingBoxes(
      const std::vector<std::vector<TriBar>>& bar_groups,
      const cv::Mat_<int32_t>& cc_labels,
      const cv::Mat_<int32_t>& cc_stats,
      const std::vector<Vector2d>& mean_vectors,
      cv::Mat_<double>* bounding_boxes) const;

  void FitBoundingBoxes(
      const std::vector<std::vector<TriBar>>& bar_groups,
      const cv::Mat_<int32_t>& cc_labels,
      const cv::Mat_<int32_t>& cc_stats,
      const std::vector<Vector2d>& mean_vectors,
      std::vector<std::vector<std::array<double, 10>>>* bounding_boxes) const;

  // Find the missing bounding box when only one of two tri-bars in a
  // horizontal/vertical pair was found.
  //
  // Arguments:
  //  bounding_boxes  Input/Output: Tri-bar bounding boxes. Will be updated with
  //                                inferred locations.
  void CompletePartialPairs(cv::Mat_<double>& bounding_boxes) const;

  // Complete the lower levels of the USAF-1951 target by using the bar spacing
  // information from the top level. As such, the upper and lower level need to
  // have at least one detected tri-bar group in common.
  //
  // Arguments:
  //  bounding_boxes  Input/Output: Tri-bar bounding boxes (no partial pairs).
  //                                Will be updated with inferred locations.
  //  level           The level to be completed
  void CompleteLowerLevel(cv::Mat_<double>& bounding_boxes, int level) const;

  // Determine whether the horizontal bars are the first group in bb_centroids
  // and the corresponding array. If not, everything should be swapped.
  //
  // Arguments:
  //  bounding_boxes  Tri-bar bounding boxes (complete)
  //  mean_vectors    The orientation vectors of the target.
  bool IsHorizontalFirst(
       const std::vector<std::vector<TriBar>>& tri_bars,
       const cv::Mat_<double>& cc_centroids) const;
  bool IsHorizontalFirst(
       const cv::Mat_<double>& bounding_boxes,
       const std::vector<Vector2d>& mean_vectors) const;

  // Get the region in which to measure a profile.
  //
  // Arguments:
  //  bar_group    The index of the bar group to measure
  //  orientation  The orientation of the bars to measure. See Orientation.
  //  region       Output: The profile bounding box
  void GetProfileRegion(int bar_group,
                        int orientation,
                        BoundingBox* region) const;

  // Test whether a point is within the given quad.
  //
  // Arguments:
  //  x     X-coordinate of test point
  //  y     Y-coordinate of test point
  //  quad  Quad to test.
  bool PointInQuad(double x, double y, const BoundingBox& quad) const;

  // Get a visualization of the connected components in the image.
  cv::Mat VisualizeCCs(const cv::Mat_<int32_t>& cc_labels) const;

  // Get a visualization of the detected bars in the image.
  cv::Mat VisualizeBars(const cv::Mat_<int32_t>& cc_labels,
                        const std::vector<int>& bar_ccs) const;

  // Get a visualization of the detected bars in the image.
  cv::Mat VisualizeTriBars(
      const cv::Mat_<int32_t>& cc_labels,
      const std::vector<std::vector<TriBar>>& tribars) const;

   template <typename T>
   static double BoxArea(const T& box) {
     return TriangleArea(box[0], box[1], box[2], box[3], box[4], box[5]) +
            TriangleArea(box[0], box[1], box[6], box[7], box[4], box[5]);
   }

   static double TriangleArea(double x1, double y1, double x2, double y2,
                              double x3, double y3) {
     return 0.5 * std::abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2));
   }

 private:
  static const std::vector<cv::Scalar> kColors;

 private:
  // The image of the target.
  cv::Mat image_;

  // The number of levels of the target are nested within each other.
  int num_levels_;

  // The bounding boxes of each tri-bar target. Even rows are horizontal
  // tri-bars and odd rows are verticals. Elements 0-7 are the (x, y)
  // coordinates of the box corners and elements 8 and 9 are the (x, y) of the
  // box's centroid.
  cv::Mat_<double> bounding_boxes_;

  // The mean vectors for horizontal and vertical tri-bar orientations.
  std::vector<Vector2d> mean_vectors_;
};

#endif  // USAF_1951_TARGET_H
