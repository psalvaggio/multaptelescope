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
DEFINE_string(output_prefix, "./", "Prefix for output files.");
DEFINE_bool(whole_image, false, "If specified, will not ask for an ROI.");
DEFINE_string(tmpl, "", "Filename of the template");

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
  string dir = ResolvePath(DirectoryName(FLAGS_output_prefix));
  if (!is_dir(dir)) {
    if (!boost::filesystem::create_directories(dir)) {
      cerr << "Output directory " << dir << " could not be created." << endl;
      return 1;
    }
  }

  // Perform recognition
  Usaf1951Target tribar(image, FLAGS_levels);

  string tmpl_filename = FLAGS_tmpl == "" ? "" : ResolvePath(FLAGS_tmpl);
  if (file_exists(tmpl_filename)) {
    mats::Usaf1951Template tmpl;
    if (!mats_io::ProtobufReader::Read(tmpl_filename, &tmpl)) {
      cerr << "Could not read template file." << endl;
      return 1;
    }
    tribar.UseTemplate(tmpl);
  } else if (!tribar.RecognizeTarget()) {
    cerr << "Failed to recognize target." << endl;
    return 1;
  }

  tribar.WriteTemplate(
      StringPrintf("%stemplate.tpl", FLAGS_output_prefix.c_str()));

  // Make two diagnostic outputs so the user can see if anything messed up
  Mat output = tribar.VisualizeBoundingBoxes();
  imwrite(StringPrintf("%sbounding_boxes.png", FLAGS_output_prefix.c_str()),
          output);

  output = tribar.VisualizeProfileRegions();
  imwrite(StringPrintf("%sprofile_regions.png", FLAGS_output_prefix.c_str()),
          output);

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
    filename = StringPrintf("%sgroup_%02d_horizontal",
                            FLAGS_output_prefix.c_str(), i);

    gp << "set output \"" << filename << ".eps\"\n"
       << "plot" << gp.file1d(profile) << "w l\n" << endl;

    ofstream horiz_ofs(filename + ".txt");
    if (!horiz_ofs.is_open()) continue;

    for (size_t j = 0; j < profile.size(); j++) {
      horiz_ofs << profile[j].first << "\t" << profile[j].second << endl;
    }

    tribar.GetProfile(i, Usaf1951Target::VERTICAL, &profile);
    filename = StringPrintf("%sgroup_%02d_vertical",
                            FLAGS_output_prefix.c_str(), i);

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
