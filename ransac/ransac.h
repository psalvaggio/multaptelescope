// Written by Philip Salvaggio (salvaggio.philip@gmail.com)
// Copyright 2012, All Rights Reserverd.
//
// This file defines a RAndom SAmple Consesus (RANSAC) framework. RANSAC is a
// useful model-fitting algorithm in the prescence of large amounts of outliers
// or noise. The algorithm is introduced in
//
// Martin A. Fischler and Robert C. Bolles (June 1981). "Random Sample
// Consensus: A Paradigm for Model Fitting with Applications to Image Analysis
// and Automated Cartography". Comm. of the ACM 24 (6): 381–395.
//
// Since RANSAC uses randomness extensively, the calling program
// should seed the random number generator prior to calling this library with:
//
// srand(time(NULL));
//
// The Ransac algorithm may then be run using:
//
// srand(time(NULL));
// ImplClass fitter;
// typename ImplClass::model_t best_model;
// std::vector<int> inliers;
// ransac::Error_t error = ransac::Ransac(
//       fitter,
//       data,
//       num_data_points,
//       minimum_data_points_for_model,
//       max_data_trials,
//       max_overall_trials,
//       best_model,
//       inliers);
//                                        
// The best model and inliers will only be populated if the results of RANSAC
// are valid. To check whether they will be populated, call
//
// ransac::RansacHasValidResults(error)
//
// If this returns true, then the best model will be returned and the calling
// program is responsible for freeing that memory.

#ifndef RANSAC_H_
#define RANSAC_H_

#include <cmath>
#include <cstdlib>
#include <list>
#include <set>
#include <string>
#include <vector>

namespace ransac {

// Error codes returned by Ransac().
enum Error_t {
  RansacSuccess,
  RansacMaxTrials,
  RansacDegenerateModel,
  RansacInvalidInput
};


// Returns whether the best_model and best_inliers parameters of Ransac() will 
// be populated. If this returns true, the calling program is responsible for
// freeing best_model.
//
// Parameters:
//   error - The error code returned from Ransac().
bool RansacHasValidResults(Error_t error);


// Returns a human-readable string representation of the error codes that can
// be returned from Ransac()
//
// Parameters:
//   error - The error code returned from Ransac().
std::string RansacErrorString(Error_t error);


// Generates a random sample of indices.
//
// Parameters:
//   num_to_select - The number of indices to be selected from the range
//                   [0, max_index)
//   max_index - The maximum index (exclusive) that can be part of the random
//               sample.
//   indices - Output vector into which to store the random sample.
void RandomSample(size_t num_to_select,
                  size_t max_index,
                  std::vector<int>* indices);


template <typename Data, typename Model>
class RansacImpl {
 public:
  using data_t = Data;
  using model_t = Model;

  // An optional pre-screening to efficiently test for model degeneracy
  //
  // Arguments:
  //  data          The data set
  //  random_sample The indices of the data points composing the random sample
  virtual bool DegeneracyScreen(const data_t& data,
                                const std::vector<int>& random_sample) const {
    return false;
  }

  // Fits model(s) to a random sample of the data set
  //
  // Arguments:
  //  data          The data set
  //  random_sample The indices of the data points composing the random sample
  //  models        Output: Append all fit models to this list. Appending no
  //                        models implies degeneracy and will result in a new
  //                        random sample being taken
  virtual void FitModel(const data_t& data,
                        const std::vector<int>& random_sample, 
                        std::vector<model_t>* models) const = 0;

  // Tests the models returned from RansacFitModel() and determines the indices
  // of the inliers in data.
  //
  // Arguments:
  //  data    The data set
  //  models  The models fit by FitModel()
  //  inliers Output: List o indices of inliers in the data set
  //
  // Returns:
  //  The index of the best model in models
  virtual int GetInliers(const data_t& data,
                         const std::vector<model_t>& models,
                         std::vector<int>* inliers) const = 0;
};


// Run the RANSAC algorithm. See the file header for a detailed description.
//
// Parameters:
//  impl - An instance of the implementation class.
//  data - The data to which to fit the model.
//  num_data_points - The number of data points in data.
//  minimum_data_points_for_model - The size of the random sample to be taken
//                                  each iteration to fit the model.
//  max_data_trials - The maximum number of trials that can be taken each
//                    iteration to fit a non-degenerate model.
//  max_overall_trials - The maximum number of RANSAC iterations to run.
//  best_model - Output parameter for the best model. The calling program takes
//               ownership if RansacHasValidResults(result) returns true.
//  best_inliers - Output parameter to hold the indices of the inliers.
template <typename Data, typename Model>
Error_t Ransac(const RansacImpl<Data, Model>& impl,
               const Data& data,
               size_t num_data_points,
               size_t minimum_data_points_for_model,
               size_t max_data_trials,
               size_t max_overall_trials,
               Model& best_model,
               std::vector<int>& best_inliers) {
  // Validate inputs.
  if (num_data_points < minimum_data_points_for_model) {
    return RansacInvalidInput;
  }

  Error_t result = RansacSuccess;

  // Initialize containers for the current iteration's models, the current
  // iteration's inliers, the best inliers and the random sample of indices.
  std::vector<Model> current_models;
  std::vector<int> current_inliers;
  current_inliers.reserve(num_data_points);
  std::vector<int> random_sample(minimum_data_points_for_model, 0);

  // p = 0.99 => A 99% probability of selecting the optimal model if the
  // terminating condition is reached. (No guarantee if max_overall_trials is
  // reached).
  double p = 0.99;

  // Keep track of the iteration number and best result.
  size_t trial_count = 0;
  size_t best_num_inliers = 0;

  // N is the upper bound on iterations to ensure the condition of p. This
  // is determined by the current value of best_num_inliers;
  size_t N = max_overall_trials;

  // Used if a termination condition is reached.
  bool status = true;

  const double EPS = 1.11e-16;

  while (N > trial_count) {
    bool degenerate = true;
    size_t data_trial = 0;
    current_models.clear();
    current_inliers.clear();

    // Get the model(s) for this iteration.
    while (degenerate) {
      // Make a random sample of size minimum_data_points_for_model.
      RandomSample(minimum_data_points_for_model,
                   num_data_points,
                   &random_sample);

      // Use the implementation's quick screen for degeneracy.
      degenerate = impl.DegeneracyScreen(data, random_sample);

      // If they didn't think it was degenerate, fit an actual model to
      // the data. There could potentially be more than one. There could also
      // be none, if they were wrong about degeneracy.
      if (!degenerate) {
        impl.FitModel(data, random_sample, &current_models);
        if (current_models.empty()) degenerate = true;
      }
      
      // Enforce the data trials threshold.
      if (data_trial >= max_data_trials) {
        result = RansacDegenerateModel;
        status = false;
        break;
      }
      data_trial++;
    }

    if (!status) break;

    // Call the implementation method to get the inliers for a given model.
    size_t current_best_index = impl.GetInliers(data,
                                                current_models,
                                                &current_inliers);
    size_t num_inliers = current_inliers.size();

    // If this model gave the best number of inliers, keep it around.
    if (num_inliers > best_num_inliers) {
      // Update our current bests.
      best_num_inliers = num_inliers;
      best_inliers.swap(current_inliers);
      std::swap(best_model, current_models[current_best_index]);

      // Update the needed number of trials.
      double fraction_of_inliers = num_inliers / double(num_data_points);
      double p_no_outliers = 1 - pow(fraction_of_inliers,
                                     minimum_data_points_for_model);
      p_no_outliers = (EPS > p_no_outliers) ? EPS : p_no_outliers;
      p_no_outliers = (1 - EPS < p_no_outliers) ? 1 - EPS : p_no_outliers;
      N = log(1-p) / log(p_no_outliers);
    }

    // Enforce the max iteration.
    trial_count++;
    if (trial_count > max_overall_trials) {
      result = RansacMaxTrials;
      status = false;
      break;
    }
  }

  return result;
}

}  // namespace ransac

#endif // RANSAC_H_
