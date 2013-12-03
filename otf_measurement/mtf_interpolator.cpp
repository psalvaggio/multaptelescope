// File Description
// Author: Philip Salvaggio

#include "mtf_interpolator.h"
#include "ransac/ransac.h"
#include "ransac/ransac_fit_circle.h"
#include <flann/flann.hpp>
#include <flann/defines.h>
#include <list>


MtfInterpolator::MtfInterpolator() {}

MtfInterpolator::~MtfInterpolator() {}

struct CompareAngles : std::binary_function<size_t, size_t, bool> {
  CompareAngles(const std::vector<double>& angles)
      : angles_(angles) {}

  bool operator()(size_t lhs, size_t rhs) const {
    return angles_[lhs] < angles_[rhs];
  }

  const std::vector<double>& angles_;
};

struct MtfSampleCircle {
  double radius;
  const std::vector<int>* indices;

  MtfSampleCircle(double r, const std::vector<int>* idx)
      : radius(r), indices(idx) {}
};

bool operator<(const MtfSampleCircle& lhs, const MtfSampleCircle& rhs) {
  return lhs.radius < rhs.radius;
}

struct SampleCirclesDistance : std::binary_function<
    const MtfSampleCircle&, const MtfSampleCircle&, double> {
  double operator()(const MtfSampleCircle& lhs, const MtfSampleCircle& rhs) const {
    return lhs.radius - rhs.radius;
  }
};

void MtfInterpolator::GetMtf(double* samples,
                             int num_samples,
                             int rows,
                             int cols,
                             bool circular_symmetry,
                             cv::Mat* mtf) {
  if (!mtf) return;

  /*
  flann::Matrix<double> data(new double[num_samples*2], num_samples, 2);
  for (int i = 0; i < num_samples; i++) {
    double r = sqrt(samples[3*i + 0]*samples[3*i + 0] +
                    samples[3*i + 1]*samples[3*i + 1]);
    double theta = atan2(samples[3*i + 1], samples[3*i + 0]);
    if (theta < -M_PI / 2) theta += M_PI;
    if (theta >= M_PI / 2) theta -= M_PI;

    *(data[2*i + 0]) = r * 2 * M_PI;
    *(data[2*i + 1]) = theta;
  }

  flann::Index<flann::L2<double> >
      knn_searcher(data, flann::KDTreeSingleIndexParams());
  knn_searcher.buildIndex();
  std::cout << "built index" << std::endl;
  */

  std::vector<double> data(2*num_samples, 0);
  std::vector<int> indices(num_samples, 0);
  std::vector<double> angles(num_samples, 0);
  for (int i = 0; i < num_samples; i++) {
    data[2*i] = samples[3*i + 0];
    data[2*i+1] = samples[3*i + 1];
    angles[i] = atan2(samples[3*i + 1], samples[3*i + 0]);
    indices[i] = i;
  }

  RansacFitOriginCircle circle_fitter(0.005);
  std::vector<double> circle_radii;
  std::vector<std::vector<int> > circles;
  while (data.size() > 0) {
    std::vector<double>* radius = NULL;
    std::list<int> inliers;
    ransac::Error_t result = ransac::Ransac(circle_fitter,
                                            data,
                                            data.size() / 2,
                                            1,
                                            1,
                                            10,
                                            &radius,
                                            &inliers);
    if (!RansacHasValidResults(result)) {
      std::cerr << "Error: Could not fit circle to MTF samples" << std::endl;
      std::cerr << "Ransac Error: " << RansacErrorString(result) << std::endl;
      return;
    }

    std::cout << "Fit a circle (r=" << (*radius)[0] << ") with "
              << inliers.size() << " points" << std::endl;
    circle_radii.push_back((*radius)[0]);
    delete radius;

    std::vector<int> inliers_vec;
    inliers_vec.insert(inliers_vec.begin(), inliers.begin(), inliers.end());
    std::sort(inliers_vec.begin(), inliers_vec.end());

    circles.push_back(std::vector<int>());
    std::vector<int>& circle_indices(circles[circles.size() - 1]);
    for (int i = 0; i < inliers_vec.size(); i++) {
      circle_indices.push_back(indices[inliers_vec[i]]);
    }

    for (int i = inliers_vec.size() - 1; i >= 0; i--) {
      std::vector<double>::iterator data_it = data.begin() + 2*inliers_vec[i];
      data.erase(data_it, data_it + 2);
      indices.erase(indices.begin() + inliers_vec[i]);
    }
  }

  for (int i = 0; i < circles.size(); i++) {
    std::sort(circles[i].begin(), circles[i].end(), CompareAngles(angles));
  }

  std::vector<MtfSampleCircle> sample_circles;
  sample_circles.reserve(circles.size());
  for (int i = 0; i < circles.size(); i++) {
    sample_circles.push_back(MtfSampleCircle(circle_radii[i], &(circles[i])));
  }
  std::sort(sample_circles.begin(), sample_circles.end());

  mtf->create(rows, cols, CV_64FC1);
  double* mtf_data = (double*)mtf->data;

  size_t half_x = cols / 2;
  size_t half_y = rows / 2;

  const int kDistancePower = 4;
  const int kAngularNeighborhood = 2;

  for (int i = 0; i < rows; i++) {
    double eta = ((double)i - half_y) / rows;
    if (circular_symmetry) eta = fabs(eta);

    for (int j = 0; j < cols; j++) {
      double xi = ((double)j - half_x) / cols;
      if (circular_symmetry) xi = fabs(xi);

      double r = sqrt(xi * xi + eta * eta);
      double theta = atan2(eta, xi);

      int next_circle;
      for (next_circle = 0; next_circle < sample_circles.size(); next_circle++) {
        if (r < sample_circles[next_circle].radius) break;
      }

      std::vector<int> nearest_circles;
      std::vector<double> circle_weights;
      if (next_circle == 0) {
        nearest_circles.push_back(next_circle);
        circle_weights.push_back(1);
      } else if (next_circle == sample_circles.size()) {
        nearest_circles.push_back(next_circle - 1);
        circle_weights.push_back(1);
      } else {
        nearest_circles.push_back(next_circle);
        nearest_circles.push_back(next_circle - 1);
        double alpha = (sample_circles[next_circle].radius - r) /
                       (sample_circles[next_circle].radius -
                        sample_circles[next_circle-1].radius);
        circle_weights.push_back(1 - alpha);
        circle_weights.push_back(alpha);
      }

      std::vector<int> nn_indices;
      std::vector<int> nn_distances;

      double mtf_val = 0;
      for (int k = 0; k < nearest_circles.size(); k++) {
        int circle = nearest_circles[k];
        const std::vector<int>& circle_pts(*(sample_circles[circle].indices));
        int next_angle;
        for (next_angle = 0; next_angle < circle_pts.size(); next_angle++) {
          if (theta < angles[circle_pts[next_angle]]) {
            break;
          }
        }

        double circle_mtf_val = 0;
        if (next_angle == 0) {
          circle_mtf_val = samples[3*circle_pts[next_angle] + 2];
        } else if (next_angle == circle_pts.size()) {
          circle_mtf_val = samples[3*circle_pts[next_angle-1] + 2];
        } else {
          double alpha = (angles[circle_pts[next_angle]] - theta) /
                         (angles[circle_pts[next_angle]] - 
                          angles[circle_pts[next_angle - 1]]);
          circle_mtf_val = (1 - alpha) * samples[3*circle_pts[next_angle] + 2] +
                           alpha * samples[3*circle_pts[next_angle-1] + 2];
        }

        mtf_val += circle_mtf_val * circle_weights[k];
      }

      mtf_data[i*cols + j] = mtf_val;

      /*
      double mtf_val = 0;
      double total_weight = 0;
      for (int k = 0; k < nn_indices.size(); k++) {
        double dxi = xi - samples[3*nn_indices[k]];
        double deta = eta - samples[3*nn_indices[k] + 1];
        double distance = sqrt(dxi*dxi + deta*deta);
        double tmp_mtf = samples[3*nn_indices[k] + 2];

        if (distance < 1e-4) distance = 1e-4;

        double weight = 1 / pow(distance, kDistancePower);
        total_weight += weight;
        mtf_val += weight * tmp_mtf;
      }
      mtf_data[i*cols + j] = mtf_val / total_weight;
      */

    }
  }
}
