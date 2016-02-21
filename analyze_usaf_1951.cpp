// Analyze a USAF-1951 tri-bar target by taking profiles over the various
// tri-bar groups.
// Author: Philip Salvaggio

#include "mats.h"

#include <boost/filesystem.hpp>
#include <gflags/gflags.h>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;
using namespace mats;

DEFINE_string(image, "", "Image filename");
DEFINE_int32(levels, 2, "Number of nested USAF targets.");
DEFINE_string(output_dir, ".", "Directory into which to store the output.");
DEFINE_bool(whole_image, false, "If specified, will not ask for an ROI.");

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);

  // Get the Tri-bar target region in the image
  Mat image;
  {
    Mat image_full = imread(ResolvePath(FLAGS_image), 0);
    if (!image_full.data) {
      cerr << "Could not read image file." << endl;
      return 1;
    }

    if (FLAGS_whole_image) {
      image_full.copyTo(image);
    } else {
      vector<uint16_t> roi = GetRoi(image_full);
      image_full(Range(roi[1], roi[1] + roi[3]),
                 Range(roi[0], roi[0] + roi[2])).copyTo(image);
    }
  }

  // Validate the output directory
  string dir = ResolvePath(FLAGS_output_dir);
  if (!is_dir(dir)) {
    if (!boost::filesystem::create_directories(dir)) {
      cerr << "Output directory " << dir << " could not be created." << endl;
      return 1;
    }
    dir = AppendSlash(dir);
  }

  // Perform recognition
  Usaf1951Target tribar(image, FLAGS_levels);
  if (!tribar.RecognizeTarget()) {
    cerr << "Failed to recognize target." << endl;
    return 1;
  }

  // Make two diagnostic outputs so the user can see if anything messed up
  Mat output = tribar.VisualizeBoundingBoxes();
  imwrite(StringPrintf("%sbounding_boxes.png", dir.c_str()), output);

  output = tribar.VisualizeProfileRegions();
  imwrite(StringPrintf("%sprofile_regions.png", dir.c_str()),
          ByteScale(output));

  // Output the profiles
  Gnuplot gp;
  gp << "set xlabel \"Pixel Location\"\n"
     << "set ylabel \"Digital Count\"\n"
     << "unset key\n"
     << "set terminal postscript eps enhanced color\n";
  for (int i = 0; i < tribar.num_bar_groups(); i++) {
    if (!tribar.FoundBarGroup(i)) continue;

    vector<pair<double, double>> profile;
    string filename;

    tribar.GetProfile(i, Usaf1951Target::HORIZONTAL, &profile);
    filename = StringPrintf("%sgroup_%02d_horizontal", dir.c_str(), i);

    gp << "set output \"" << filename << ".eps\"\n"
       << "plot" << gp.file1d(profile) << "w l\n" << endl;

    ofstream horiz_ofs(filename + ".txt");
    if (!horiz_ofs.is_open()) continue;

    for (size_t j = 0; j < profile.size(); j++) {
      horiz_ofs << profile[j].first << "\t" << profile[j].second << endl;
    }

    tribar.GetProfile(i, Usaf1951Target::VERTICAL, &profile);
    filename = StringPrintf("%s/group_%02d_vertical", dir.c_str(), i);

    gp << "set output \"" << filename << ".eps\"\n"
       << "plot" << gp.file1d(profile) << "w l\n" << endl;

    ofstream vert_ofs(filename + ".txt");
    if (!vert_ofs.is_open()) continue;

    for (size_t j = 0; j < profile.size(); j++) {
      vert_ofs << profile[j].first << "\t" << profile[j].second << endl;
    }
  }

  return 0;
}
