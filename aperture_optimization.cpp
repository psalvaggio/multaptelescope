// File Description
// Author: Philip Salvaggio

#include "genetic/genetic_algorithm.h"
#include "base/pupil_function.h"
#include "optical_designs/compound_aperture.h"

//#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "base/opencv_utils.h"

#include <cmath>
#include <fstream>

static const int kNumPoints = 6;
static const double kEncircledDiameter = 3;
static const double kSubapertureDiameter = 0.625;
static const double kSamplesInDiameter = 512;

static const int kPopulationSize = 8;
static const double kCrossoverProbability = 0;
static const double kMutateProbability = 1;
static const int kBreedsPerGeneration = 4;
static const double kIntroduceProbability = 0.01;

static mats::SimulationConfig conf;

using namespace std;
using namespace genetic;

class MomentOfInertiaImpl : public GeneticAlgorithmImpl<vector<double>> {
 public:
  MomentOfInertiaImpl(int num_subapertures,
              double max_center_radius,
              double subaperture_diameter,
              double sample_step,
              double mutate_probability,
              double crossover_probability)
      : num_subapertures_(num_subapertures),
        max_center_radius2_(max_center_radius*max_center_radius),
        subaperture_diameter2_(subaperture_diameter*subaperture_diameter),
        sample_step_(sample_step),
        should_continue_(true),
        mutate_probability_(mutate_probability),
        crossover_probability_(crossover_probability) {}

  static void Destroy(model_t& model) {
    (void) model;
  }

  bool Evaluate(PopulationMember<model_t>& member);

  model_t Introduce();

  model_t Crossover(const PopulationMember<model_t>& member1,
                    const PopulationMember<model_t>& member2);

  void Mutate(PopulationMember<model_t>& member);

  bool ShouldContinue(const vector<PopulationMember<model_t>>& population,
                      size_t generation_num) {
    (void) population; (void) generation_num;
    return should_continue_;
  }

  void Stop() { should_continue_ = false; }

  void GetAutocorrelationPeaks(const model_t& locations, model_t* peaks);

  void ZeroMean(model_t* locations);

  void Visualize(const model_t& locations);

 private:
  int num_subapertures_;
  double max_center_radius2_;
  double subaperture_diameter2_;
  double sample_step_;
  bool should_continue_;
  double mutate_probability_;
  double crossover_probability_;
};

bool MomentOfInertiaImpl::Evaluate(PopulationMember<model_t>& member) {
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
      conf.mutable_simulation(0)->mutable_aperture_params());
  CompoundApertureParameters* compound_params =
      ap_params->MutableExtension(compound_aperture_params);
  for (size_t i = 0; i < locations.size(); i += 2) {
    mats::ApertureParameters* subap = compound_params->mutable_aperture(i/2);
    subap->set_offset_x(locations[i]);
    subap->set_offset_y(locations[i+1]);
  }

  mats::PupilFunction pupil;
  unique_ptr<Aperture> aperture(new CompoundAperture(conf, 0));
  aperture->GetPupilFunction(aperture->GetWavefrontError(), 550e-9, &pupil);
  cv::Mat mtf = FFTShift(pupil.ModulationTransferFunction());
  double enc_diameter = aperture->encircled_diameter();

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
  //cv::imshow("MTF", GammaScale(mtf, 1/2.2));
  //cv::imshow("Mask", ByteScale(aperture->GetApertureMask()));
  //cv::waitKey(1);
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

MomentOfInertiaImpl::model_t MomentOfInertiaImpl::Introduce() {
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

MomentOfInertiaImpl::model_t MomentOfInertiaImpl::Crossover(
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

void MomentOfInertiaImpl::Mutate(PopulationMember<model_t>& member) {
  model_t& locations(member.model());

  for (size_t i = 0; i < locations.size(); i += 2) {
    double p = (double)rand() / RAND_MAX;
    if (p < mutate_probability_) {
      /*
      int r_dir = (rand() % 11) - 5;
      int theta_dir = (rand() % 11) - 5;

      double new_r = sqrt(pow(locations[i], 2) + pow(locations[i+1], 2)) +
                     r_dir * sample_step_;
      double new_theta = atan2(locations[i+1], locations[i]) +
                        theta_dir * sample_step_;
                        */
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

void MomentOfInertiaImpl::ZeroMean(model_t* locations) {
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



void MomentOfInertiaImpl::GetAutocorrelationPeaks(const model_t& locations,
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

static MomentOfInertiaImpl impl(
    kNumPoints,
    0.5 * (kEncircledDiameter - kSubapertureDiameter),
    kSubapertureDiameter,
    kEncircledDiameter / kSamplesInDiameter,
    kMutateProbability,
    kCrossoverProbability);

static bool has_stopped = false;
void stop_iteration(int signo) {
  (void) signo;
  if (has_stopped) {
    exit(1);
  } else {
    impl.Stop();
    has_stopped = true;
  }
}

void MomentOfInertiaImpl::Visualize(const model_t& locations) {
  /*
  mats::ApertureParameters* ap_params(
      conf.mutable_simulation(0)->mutable_aperture_params());
  CompoundApertureParameters* compound_params =
      ap_params->MutableExtension(compound_aperture_params);
  for (size_t i = 0; i < locations.size(); i += 2) {
    mats::ApertureParameters* subap = compound_params->mutable_aperture(i/2);
    subap->set_offset_x(locations[i]);
    subap->set_offset_y(locations[i+1]);
  }

  mats::PupilFunction pupil;
  unique_ptr<Aperture> aperture(new CompoundAperture(conf, 0));
  aperture->GetPupilFunction(aperture->GetWavefrontError(), 550e-9, &pupil);
  cv::Mat mtf = pupil.ModulationTransferFunction();
  cv::imshow("Best MTF", FFTShift(GammaScale(mtf, 1/2.2)));
  cv::imshow("Best Mask", ByteScale(aperture->GetApertureMask()));
  cv::waitKey(1);
  */
}


int main() {
  signal(SIGINT, stop_iteration);
  srand(time(NULL));

  /*
  cv::namedWindow("MTF", cv::WINDOW_AUTOSIZE);
  cv::moveWindow("MTF", 0, 0);
  cv::namedWindow("Mask", cv::WINDOW_AUTOSIZE);
  cv::moveWindow("Mask", 0, 600);
  cv::namedWindow("Best MTF", cv::WINDOW_AUTOSIZE);
  cv::moveWindow("Best MTF", 600, 0);
  cv::namedWindow("Best Mask", cv::WINDOW_AUTOSIZE);
  cv::moveWindow("Best Mask", 600, 600);
  */

  conf.set_array_size(512);
  conf.set_reference_wavelength(550e-9);
  mats::Simulation* sim = conf.add_simulation();
  mats::ApertureParameters* compound_params = sim->mutable_aperture_params();
  compound_params->set_encircled_diameter(kEncircledDiameter);
  compound_params->set_type(mats::ApertureParameters::COMPOUND);
  CompoundApertureParameters* array_ext =
      compound_params->MutableExtension(compound_aperture_params);
  array_ext->set_combine_operation(CompoundApertureParameters::OR);
  for (size_t i = 0; i < kNumPoints; i++) {
    mats::ApertureParameters* subap = array_ext->add_aperture();
    subap->set_type(mats::ApertureParameters::CIRCULAR);
    subap->set_encircled_diameter(kSubapertureDiameter);
    subap->set_offset_x(0);
    subap->set_offset_y(0);
  }

  typename MomentOfInertiaImpl::model_t best_locations;
  GeneticAlgorithm(impl, 
                   kPopulationSize,
                   kBreedsPerGeneration,
                   best_locations);

  ofstream ofs("locations.txt");
  ofs << "set parametric" << endl
      << "unset key" << endl
      << "set angle degree" << endl
      << "set size square" << endl
      << "set trange [0:360]" << endl
      << "r = " << kEncircledDiameter * 0.5 << endl
      << "r2 = " << kSubapertureDiameter * 0.5 << endl
      << "plot \"-\" u 1:2, r*cos(t), r*sin(t)";

  for (size_t i = 0; i < best_locations.size(); i += 2) {
    ofs << ", r2*cos(t)";

    if (best_locations[i] >= 0) {
      ofs << " + " << best_locations[i];
    } else {
      ofs << best_locations[i];
    }
    ofs << ",r2*sin(t)";
    if (best_locations[i+1] >= 0) {
      ofs << " + " << best_locations[i+1];
    } else {
      ofs << best_locations[i+1];
    }
  }
  ofs << endl;
  for (size_t i = 0; i < best_locations.size(); i += 2) {
    ofs << best_locations[i] << "\t" << best_locations[i+1] << endl;
  }

  return 0;
}
