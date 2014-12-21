// File Description
// Author: Philip Salvaggio

#include "golay_genetic_impl.h"
#include "base/pupil_function.h"
#include "base/opencv_utils.h"
#include "optical_designs/compound_aperture.h"

#include <opencv/cv.h>
#include <opencv/highgui.h>

using namespace std;
using namespace genetic;

GolayGeneticImpl::GolayGeneticImpl(int num_subapertures,
                                   double max_center_radius,
                                   double subaperture_diameter,
                                   double mutate_probability,
                                   double crossover_probability)
      : num_subapertures_(num_subapertures),
        max_center_radius2_(max_center_radius*max_center_radius),
        subaperture_diameter2_(subaperture_diameter*subaperture_diameter),
        should_continue_(true),
        mutate_probability_(mutate_probability),
        crossover_probability_(crossover_probability),
        conf_() {
  conf_.set_array_size(512);
  conf_.set_reference_wavelength(550e-9);

  mats::Simulation* sim = conf_.add_simulation();
  mats::ApertureParameters* compound_params = sim->mutable_aperture_params();

  double enc_diameter = 2 * max_center_radius + subaperture_diameter;
  cout << "Diameter: " << enc_diameter << endl;
  compound_params->set_encircled_diameter(enc_diameter);

  compound_params->set_type(mats::ApertureParameters::COMPOUND);
  CompoundApertureParameters* array_ext =
      compound_params->MutableExtension(compound_aperture_params);
  array_ext->set_combine_operation(CompoundApertureParameters::OR);
  for (int i = 0; i < num_subapertures; i++) {
    mats::ApertureParameters* subap = array_ext->add_aperture();
    subap->set_type(mats::ApertureParameters::CIRCULAR);
    subap->set_encircled_diameter(subaperture_diameter);
    subap->set_offset_x(0);
    subap->set_offset_y(0);
  }
}


bool GolayGeneticImpl::Evaluate(PopulationMember<model_t>& member) {
  double moment_of_inertia = 0;

  const model_t& locations(member.model());

  // Check for non-overlapping subapertures
  for (size_t i = 0; i < locations.size(); i += 2) {
    double sq_dist = locations[i]*locations[i] + locations[i+1]*locations[i+1];
    if (sq_dist > max_center_radius2_) return false;

    for (size_t j = i + 2; j < locations.size(); j += 2) {
      double dx = locations[i] - locations[j];
      double dy = locations[i+1] - locations[j+1];
      double distance2 = dx*dx + dy*dy;
      if (distance2 < subaperture_diameter2_)  return false;
    }
  }

  mats::ApertureParameters* ap_params(
      conf_.mutable_simulation(0)->mutable_aperture_params());
  ap_params->set_type(mats::ApertureParameters::COMPOUND);
  CompoundApertureParameters* compound_params =
      ap_params->MutableExtension(compound_aperture_params);
  for (size_t i = 0; i < locations.size(); i += 2) {
    mats::ApertureParameters* subap = compound_params->mutable_aperture(i/2);
    subap->set_offset_x(locations[i]);
    subap->set_offset_y(locations[i+1]);
  }

  mats::PupilFunction pupil;
  unique_ptr<Aperture> aperture(ApertureFactory::Create(conf_, 0));
  aperture->GetPupilFunction(aperture->GetWavefrontError(), 550e-9, &pupil);
  cv::Mat mtf = FFTShift(pupil.ModulationTransferFunction());
  //double enc_diameter = aperture->encircled_diameter();

  /*
  double fitness = 0;
  std::vector<double> angles{0, 30, 60, 90, 120, 150};
  for (double angle : angles) {
    std::vector<double> profile;
    GetRadialProfile(mtf, angle * M_PI / 180, &profile);

    int last_significant_index = 0;
    for (int i = profile.size() - 1; i >= 0; i--) {
      if (profile[i] > 1e-3) {
        last_significant_index = i;
        break;
      }
    }
    double diameter = enc_diameter *
                      (double)last_significant_index / profile.size();

    double smoothness = 0;
    for (size_t i = 1; i <= last_significant_index; i++) {
      smoothness += fabs(profile[i-1] - profile[i]);
    }
    smoothness = 1 - smoothness;

    fitness += smoothness + 0.5 * diameter;
    //fitness += diameter;
  }
  */
  cv::imshow("MTF", GammaScale(mtf, 1/2.2));
  cv::imshow("Mask", ByteScale(aperture->GetApertureMask()));
  cv::waitKey(1);
  //cv::Scalar sum = cv::sum(mtf);
  //double diameter = aperture->encircled_diameter();
  //double fitness = diameter + 1e-4 * sum[0] / (M_PI * diameter * diameter / 4);
  //member.set_fitness(fitness);
  //return true;



  model_t peaks;
  GetAutocorrelationPeaks(member.model(), &peaks);
  for (size_t i = 0; i < peaks.size(); i += 2) {
    moment_of_inertia += peaks[i]*peaks[i] + peaks[i+1]*peaks[i+1];
  }
  double compactness = 1 / moment_of_inertia;
  cv::Mat mtf_float;
  mtf.convertTo(mtf_float, CV_32F);
  cv::threshold(mtf_float, mtf_float, 0.05, 1, cv::THRESH_TOZERO);
  int non_zeros = cv::countNonZero(mtf_float);
  double support_frac = non_zeros / (M_PI * pow(mtf.cols / 2, 2));
  double fitness = compactness + 10 * support_frac;

  member.set_fitness(fitness);

  return true;
}

GolayGeneticImpl::model_t GolayGeneticImpl::Introduce() {
  model_t tmp_model;
  PopulationMember<model_t> member(std::move(tmp_model));
  model_t& locations(member.model());
  locations.resize(2*num_subapertures_, 0);

  bool keep_going = true;

  while (keep_going) {
    for (int i = 0; i < num_subapertures_; i++) {
      double r = (double)rand() / RAND_MAX * sqrt(max_center_radius2_);
      double theta = (double)rand() / RAND_MAX * 2 * M_PI;
      locations[2*i] = r * cos(theta);
      locations[2*i + 1] = r * sin(theta);
    }
    keep_going = !Evaluate(member);
  }

  model_t new_model = std::move(member.model());
  ZeroMean(&new_model);
  return new_model;
}

GolayGeneticImpl::model_t GolayGeneticImpl::Crossover(
    const PopulationMember<model_t>& member1,
    const PopulationMember<model_t>& member2) {
  const model_t& input1_locs(member1.model());
  const model_t& input2_locs(member2.model());
  model_t output_locs;
  output_locs.resize(2*num_subapertures_, 0);

  for (size_t i = 0; i < input1_locs.size(); i += 2) {
    double p = (double)rand() / RAND_MAX;
    if (p < crossover_probability_) {
      output_locs[i] = input2_locs[i];
      output_locs[i+1] = input2_locs[i+1];
    } else {
      output_locs[i] = input1_locs[i];
      output_locs[i+1] = input1_locs[i+1];
    }
  }

  return output_locs;
}

void GolayGeneticImpl::Mutate(PopulationMember<model_t>& member) {
  model_t& locations(member.model());

  for (size_t i = 0; i < locations.size(); i += 2) {
    double p = (double)rand() / RAND_MAX;
    if (p < mutate_probability_) {
      double new_r = sqrt(max_center_radius2_) * (double)rand() / RAND_MAX;
      double new_theta = 2 * M_PI * (double)rand() / RAND_MAX;
      double new_x = new_r * cos(new_theta);
      double new_y = new_r * sin(new_theta);

      if (new_r * new_r < max_center_radius2_) {
        locations[i] = new_x;
        locations[i+1] = new_y;
      }
    }
  }

  ZeroMean(&locations);
}

void GolayGeneticImpl::ZeroMean(model_t* locations) {
  if (!locations) return;

  double mean_x = 0;
  double mean_y = 0;
  for (size_t i = 0; i < locations->size(); i += 2) {
    mean_x += (*locations)[i];
    mean_y += (*locations)[i+1];
  }
  mean_x /= locations->size() / 2;
  mean_y /= locations->size() / 2;

  for (size_t i = 0; i < locations->size(); i += 2) {
    (*locations)[i] -= mean_x;
    (*locations)[i+1] -= mean_y;
  }
}



void GolayGeneticImpl::GetAutocorrelationPeaks(const model_t& locations,
                                                  model_t* peaks) {
  if (!peaks) return;
  peaks->resize(locations.size() * (locations.size() - 1));

  int index = 0;
  for (size_t i = 0; i < locations.size(); i += 2) {
    for (size_t j = 0; j < locations.size(); j += 2) {
      if (i == j) continue;

      (*peaks)[index++] = locations[i] - locations[j];
      (*peaks)[index++] = locations[i+1] - locations[j+1];
    }
  }
}

void GolayGeneticImpl::Visualize(const model_t& locations) {
  mats::ApertureParameters* ap_params(
      conf_.mutable_simulation(0)->mutable_aperture_params());
  CompoundApertureParameters* compound_params =
      ap_params->MutableExtension(compound_aperture_params);
  for (size_t i = 0; i < locations.size(); i += 2) {
    mats::ApertureParameters* subap = compound_params->mutable_aperture(i/2);
    subap->set_offset_x(locations[i]);
    subap->set_offset_y(locations[i+1]);
  }

  mats::PupilFunction pupil;
  unique_ptr<Aperture> aperture(new CompoundAperture(conf_, 0));
  aperture->GetPupilFunction(aperture->GetWavefrontError(), 550e-9, &pupil);
  cv::Mat mtf = pupil.ModulationTransferFunction();
  cv::imshow("Best MTF", FFTShift(GammaScale(mtf, 1/2.2)));
  cv::imshow("Best Mask", ByteScale(aperture->GetApertureMask()));
  cv::waitKey(1);
}
