// Aligns a set of measured MTF profiles with the model's prediction of the
// theoretical MTF. It is assumed that each MTF profile is in the first two
// columns of a file called mtf.txt in subdirectories of the base directory
// given on the command line. The name of each subdirectory should be an initial
// estimate of the angle of the profile, in degrees. For instance, an example
// directory structure would be,
//
// base_dir/
//   00/
//     image00.png
//     ...
//     mtf.txt
//   10/
//     image00.png
//     ...
//     mtf.txt
//   ...
//
// This profram will write a text file called orientation.txt in each
// subdirectory with the aligned orientation.
//
// Author: Philip Salvaggio

#include "mats.h"

#include <csignal>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <queue>
#include <unistd.h>

#include <gflags/gflags.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>

using namespace cv;
using namespace std;

using MtfProfile = tuple<double, vector<vector<double>>, string>;

DEFINE_double(global_resolution, 2, "Search resolution for global alignment "
                                    "[deg]");
DEFINE_double(local_limit, 1, "Maximum deviation for a single profile [deg]");
DEFINE_double(local_resolution, 0.2, "Search resolution on orientation [deg]");
DEFINE_int32(reflection, 6, "The degree of angular symmetry (periodicity) in "
                            "the MTF.");

// Accessors for the angle of a profile.
const double& Angle(const MtfProfile& profile) { return get<0>(profile); }
double& Angle(MtfProfile& profile) { return get<0>(profile); }

// Accessors for profile information.
const vector<vector<double>>& Profile(const MtfProfile& profile) {
  return get<1>(profile);
}
vector<vector<double>>& Profile(MtfProfile& profile) {
  return get<1>(profile);
}

// Accessors for the file path of a profile.
string& Path(MtfProfile& profile) { return get<2>(profile); }
const string& Path(const MtfProfile& profile) { return get<2>(profile); }

// Bound an angle between 0 and the first reflection barrier
// (360 / FLAGS_reflection).
double AngleBound(double theta) {
  const double kMaxAngle = 360. / FLAGS_reflection;
  while (theta >= kMaxAngle) theta -= kMaxAngle;
  while (theta < 0) theta += kMaxAngle;
  return theta;
}

// Take the RMS error between two MTF profiles. The profiles must be the same
// size and are assumed to have the same frequency axis.
double MtfRms(const vector<double>& measured,
              const vector<double>& theoretical);

// Read in the measured profiles. It is assumed that the profiles are stored in
// the first two columns of a file called mtf.txt in subdirectores of base_dir,
// where the subdirectory name is the inital estimate of the angle.
void ReadProfiles(const string& base_dir, vector<MtfProfile>* profiles);

// Get the theoretical MTF from a SimulationConfig file.
Mat GetTheoreticalMtf(const string& config_file);

// Get profiles of the theoretical MTF at the desired angular spacing.
void GetTheoreticalProfiles(const Mat& theoretical_mtf,
                            double resolution,
                            vector<MtfProfile>* profiles);

// Run the golabl alignment.
double RunGlobalAlignment(const Mat& theoretical_mtf,
                          const vector<MtfProfile>& profiles);


int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  if (argc < 2) {
   cerr << "Usage: " << argv[0] << " config_file base_directory [flags]"
         << endl;
    return 1;
  }

  // Validate input
  CHECK(FLAGS_global_resolution >= 0);
  CHECK(FLAGS_local_limit >= 0);
  CHECK(FLAGS_local_resolution > 0);
  FLAGS_reflection = max(FLAGS_reflection, 1);

  // Read in the measured profiles
  string base_dir = mats::ResolvePath(argv[2]);
  vector<MtfProfile> profiles;
  ReadProfiles(base_dir, &profiles);
  
  // Get the theoretical MTF
  Mat mtf_2d = GetTheoreticalMtf(mats::ResolvePath(argv[1]));

  // Run the global alignment
  cout << "Running global alignment..." << endl;
  double global_offset = RunGlobalAlignment(mtf_2d, profiles);
  cout << "Global Alignment Done (offset = " << global_offset << " [deg])"
       << endl;
  
  Gnuplot gp;
  gp << "set xlabel \"Spatial Frequency [cyc / pixel]\"\n"
     << "set ylabel \"MTF\"\n"
     << "set xrange [0:0.5]\n"
     << "set yrange [0:1]\n"
     << "unset key\n";

  const int kSearchRadius = round(FLAGS_local_limit / FLAGS_local_resolution);

  // Run the local alignment
  double average_rms = 0;
  for (const auto& profile : profiles) {
    // Scan over the local neighborhood to find the minimum RMS error
    double min_rms = 1e10, min_angle = 0;
    for (int i = -kSearchRadius; i <= kSearchRadius; i++) {
      double theta = AngleBound(Angle(profile) + global_offset +
                                i * FLAGS_local_resolution);

      // Interpolate the theoretical MTF to the measured resolution.
      vector<double> raw_profile, ref_profile, theoretical_freq;
      GetRadialProfile(mtf_2d, theta * M_PI / 180, &raw_profile);
      for (size_t j = 0; j < raw_profile.size(); j++) {
        theoretical_freq.push_back(0.5 * j / (raw_profile.size() - 1));
      }
      mats::LinearInterpolator::Interpolate(theoretical_freq, raw_profile,
                                            Profile(profile)[0], &ref_profile);

      // Compute RMS error.
      double rms = MtfRms(Profile(profile)[1], ref_profile);
      if (rms < min_rms) {
        min_rms = rms;
        min_angle = theta;
      }
    }
    average_rms += min_rms;

    // Record the found orienation.
    ofstream ofs(mats::AppendSlash(Path(profile)) + "orientation.txt");
    ofs << min_angle;

    // Print out the results.
    cout << "Orientation "
         << mats::StringPrintf("%.1f", Angle(profile) + global_offset)
         << " -> " << mats::StringPrintf("%.1f", min_angle) << " (RMS = "
         << min_rms << ")" << endl;
  }

  cout << "Average RMS = " << average_rms / profiles.size() << endl;

  return 0;
}


double MtfRms(const vector<double>& measured,
              const vector<double>& theoretical) {
  int count = 0;
  double sq_error = 0;
  for (size_t i = 0; i < theoretical.size(); i++) {
    if (theoretical[i] < 0.01) continue;
    sq_error += pow(theoretical[i] - measured[i], 2);
    count++;
  }
  return sqrt(sq_error / count);
}


void ReadProfiles(const string& base_dir, vector<MtfProfile>* profiles) {
  vector<string> files;
  mats::scandir(base_dir, "", &files);

  // Read in the measured profiles.
  for (const auto& file : files) {
    string path = base_dir + file;
    if (file[0] == '.' || !mats::is_dir(path)) continue;
    
    string mtf_file = mats::AppendSlash(path) + "mtf.txt";
    if (!mats::file_exists(mtf_file)) continue;

    // Get the orientation estimate from the directory name.
    double theta = AngleBound(atof(file.c_str()));

    cout << "Detected profile in " << file << " (" << theta
         << " [deg])" << endl;

    profiles->emplace_back(theta, vector<vector<double>>(), path);

    // Read in the measured MTF.
    vector<vector<double>> raw_mtf_data;
    mats_io::TextFileReader::Parse(mtf_file, &(Profile(profiles->back())));
  }

  CHECK(profiles->size() > 0, "No profiles detected.");

  // Sort the profiles.
  sort(begin(*profiles), end(*profiles),
      [] (const MtfProfile& a, const MtfProfile& b) {
        return Angle(a) < Angle(b);
      });
}


Mat GetTheoreticalMtf(const string& config_file) {
  // Initialize the model.
  mats::SimulationConfig sim_config;
  mats::DetectorParameters det_params;
  CHECK(mats::MatsInit(config_file,
                       &sim_config,
                       &det_params,
                       nullptr, nullptr));

  // Set up the spectral resolution of the simulation.
  vector<vector<double>> raw_weighting;
  mats_io::TextFileReader::Parse(
      sim_config.spectral_weighting_filename(),
      &raw_weighting);
  const vector<double>& wavelengths(raw_weighting[0]);
  const vector<double>& spectral_weighting(raw_weighting[1]);

  // Create the telescope.
  mats::Telescope telescope(sim_config, 0, det_params);
  telescope.detector()->set_rows(512);
  telescope.detector()->set_cols(512);

  // Compute the theoretical 2D OTF of the telescope.
  Mat theoretical_otf;
  telescope.ComputeEffectiveOtf(wavelengths,
                                spectral_weighting,
                                &theoretical_otf);
  return magnitude(FFTShift(theoretical_otf));
}

void GetTheoreticalProfiles(const Mat& theoretical_mtf,
                            double resolution,
                            vector<MtfProfile>* profiles) {
  // Take the profiles.
  const double kMaxAngle = 360. / FLAGS_reflection;
  for (double theta = 0; theta < kMaxAngle; theta += resolution) {
    profiles->emplace_back(theta, vector<vector<double>>(), "");
    auto& tmp_profile = Profile(profiles->back());
    tmp_profile.emplace_back();
    tmp_profile.emplace_back();
    GetRadialProfile(theoretical_mtf, theta * M_PI / 180,
        &(tmp_profile.back()));
    for (size_t i = 0; i < tmp_profile.back().size(); i++) {
      tmp_profile.front().push_back(
          i / (2. * (tmp_profile.back().size() - 1)));
    }
  }
}

double RunGlobalAlignment(const Mat& theoretical_mtf,
                          const vector<MtfProfile>& profiles) {
  // Get the theoretical profiles at the gloval resolution
  vector<MtfProfile> ref_profiles;
  GetTheoreticalProfiles(theoretical_mtf,
      FLAGS_global_resolution, &ref_profiles);

  // Run the global alignment.
  vector<pair<double, double>> global_rms;
  double min_error = 1e10, global_offset = 0;
  for (size_t offset_idx = 0; offset_idx < ref_profiles.size(); offset_idx++) {
    double offset_deg = Angle(ref_profiles[offset_idx]) - Angle(profiles[0]);

    double average_rms = 0;

    // Compute the RMS error
    for (size_t i = 0; i < profiles.size(); i++) {
      double theta = AngleBound(Angle(profiles[i]) + offset_deg);

      size_t ref_idx = round(theta / FLAGS_global_resolution);
      if (ref_idx >= ref_profiles.size()) {
        ref_idx = ref_idx % ref_profiles.size();
      }

      // Resample theoretical profile.
      vector<double> tmp_ref_profile;
      mats::LinearInterpolator::Interpolate(
          Profile(ref_profiles[ref_idx])[0],
          Profile(ref_profiles[ref_idx])[1],
          Profile(profiles[i])[0],
          &tmp_ref_profile);

      double rms = MtfRms(Profile(profiles[i])[1], tmp_ref_profile);
      average_rms += rms;
    }

    average_rms /= profiles.size();
    global_rms.emplace_back(offset_deg, average_rms);
    if (average_rms < min_error) {
      min_error = average_rms;
      global_offset = offset_deg;
    }
  }

  Gnuplot global_gp;
  global_gp << "set xlabel \"Offset Angle [deg]\"\n"
            << "set ylabel \"Average RMS [MTF units]\"\n"
            << "unset key\n"
            << "plot" << global_gp.file1d(global_rms) << "w l\n"
            << endl;

  return global_offset;
}
