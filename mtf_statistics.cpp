// Program to compute the MTF statistics from a directory of slant edge images.
// Output:
//  Tab separated data. Fields:
//   Frequency [cyc/pixel]
//   Average MTF value
//   Standard Deviation of MTF value
//   Min MTF Value
//   Max MTF Value
//   Number of images
// Author: Philip Salvaggio

#include "mats.h"
#include "base/statistics.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

#include <gflags/gflags.h>

using namespace cv;
using namespace std;

DEFINE_string(extension, "png", "Extension of the images");
DEFINE_bool(recursive, false, "Whether to go into subdirectories.");
DEFINE_bool(global_roi, false, "Use one ROI for all directories.");
DEFINE_bool(whole_image, false, "Whether to use the entire image.");
DEFINE_string(output_filename, "", "Filename for saving results.");
DEFINE_bool(quiet, false, "Do not output plots.");

static vector<uint16_t> global_bounds;

bool AnalyzeDirectory(const string& dir) {
  // Get the images in the image directory.
  vector<string> filenames;
  mats::scandir(dir, "." + FLAGS_extension, &filenames);
  if (filenames.empty()) {
    cerr << "Could not find any images in directory " << dir << endl;
    return 1;
  }

  // Perform slant-edge MTF on each image. Use the same ROI on each image.
  SlantEdgeMtf mtf_analyzer;
  vector<vector<double>> mtfs;
  vector<uint16_t> local_bounds;
  vector<uint16_t>& bounds(FLAGS_global_roi ? global_bounds : local_bounds);
  for (const auto& filename : filenames) {
    string path = mats::AppendSlash(dir) + filename;

    Mat image = imread(path, 0);
    if (!image.data) {
      cerr << "Could not read image file." << endl;
      return 1;
    }

    Mat roi;
    if (FLAGS_whole_image) {
      roi = image;
    } else {
      if (bounds.empty()) bounds = GetRoi(image);

      if (bounds[1] + bounds[3] > image.rows ||
          bounds[0] + bounds[2] > image.cols) {
        cerr << "Warning: Skipping " << filename << " due to insufficient size."
             << endl;
        continue;
      }

      image(Range(bounds[1], bounds[1] + bounds[3]),
            Range(bounds[0], bounds[0] + bounds[2])).copyTo(roi);
    }

    double orientation;
    mtfs.emplace_back();

    mtf_analyzer.Analyze(roi, &orientation, &(mtfs.back()));
  }

  if (mtfs.size() == 0) {
    cerr << "Warning: Skipping directory " << dir << endl;
    return false;
  }

  // Verify that all of the MTF's are the same size.
  size_t size = mtfs[0].size();
  for (size_t i = 1; i < mtfs.size(); i++) {
    if (mtfs[i].size() != size) {
      cerr << "Not all MTFs reconstructed at the same resolution." << endl;
      return 1;
    }
  }

  // Calculate the MTF statistics.
  vector<double> avg_mtf(size, 0), stddev_mtf(size, 0),
                 min_mtf(size, 1), max_mtf(size, 0);
  for (size_t i = 0; i < mtfs.size(); i++) {
    for (size_t j = 0; j < size; j++) {
      double mtf_val = mtfs[i][j];
      avg_mtf[j] += mtf_val;
      stddev_mtf[j] += mtf_val * mtf_val;
      min_mtf[j] = min(min_mtf[j], mtf_val);
      max_mtf[j] = max(max_mtf[j], mtf_val);
    }
  }
  for (size_t i = 0; i < avg_mtf.size(); i++) {
    avg_mtf[i] /= mtfs.size();
    stddev_mtf[i] = sqrt(stddev_mtf[i] / mtfs.size() - pow(avg_mtf[i], 2));
  }

  // Create curves to plot and print out the statistics.
  vector<pair<double, double>> mtf_data;
  vector<tuple<double, double, double>> error_bounds;
  ofstream ofs;
  if (FLAGS_output_filename != "") {
    ofs.open(mats::AppendSlash(dir) + FLAGS_output_filename);
    if (!ofs.is_open()) {
      cerr << "Could not open output file." << endl;
    }
  }
  ostream& os(FLAGS_output_filename != "" ? ofs : cout);

  for (size_t i = 0; i < avg_mtf.size(); i++) {
    double freq = i / (2. * (avg_mtf.size() - 1));
    mtf_data.emplace_back(freq, avg_mtf[i]);

    error_bounds.emplace_back(freq, min_mtf[i], max_mtf[i]);
    os << freq << "\t" << avg_mtf[i] << "\t" << stddev_mtf[i] << "\t"
       << min_mtf[i] << "\t" << max_mtf[i] << "\t" << mtfs.size() << endl;
  }

  if (FLAGS_quiet) return true;

  // Plot the average and the bounds
  static Gnuplot gp;
  gp << "set xlabel \"Spatial Frequency [cyc/pixel]\"\n"
     << "set ylabel \"MTF\"\n"
     << "set yrange [0:1]\n"
     << "set style fill solid 1\n" 
     << "unset key\n"
     << "plot " << gp.file1d(error_bounds) << "u 1:2:3 w filledcurves ls 2,"
     << gp.file1d(mtf_data) << " w l lw 3 lc 1\n"
     << endl;

  // Plot the individual curves.
  static Gnuplot gp2;
  gp2 << "set xlabel \"Spatial Frequency [cyc/pixel]\"\n"
      << "set ylabel \"MTF\"\n"
      << "set yrange [0:1]\n"
      << "unset key\n"
      << "plot";
  for (size_t i = 0; i < mtfs.size(); i++) {
    vector<pair<double, double>> mtf_data;
    for (size_t j = 0; j < avg_mtf.size(); j++) {
      double freq = j / (2. * (avg_mtf.size() - 1));
      mtf_data.emplace_back(freq, mtfs[i][j]);
    }
    gp2 << gp2.file1d(mtf_data) << " w l lw 2";
    if (i != mtfs.size() - 1) gp2 << ",";
  }
  gp2 << endl;

  return true;
}

void RecursiveAnalyze(const string& dir) {
  AnalyzeDirectory(dir);
  vector<string> subdirs;
  mats::subdirectories(dir, &subdirs);
  for (const auto& subdir : subdirs) {
    RecursiveAnalyze(mats::AppendSlash(dir) + subdir);
  }
}

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  if (argc < 2) {
    cerr << "Usage: " << argv[0] << " image_dir" << endl;
    return 1;
  }

  string directory(mats::ResolvePath(argv[1]));

  if (!FLAGS_recursive) {
    return AnalyzeDirectory(directory) ? 0 : 1;
  } else {
    RecursiveAnalyze(directory);
  }
  
  return 0;
}
