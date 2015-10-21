// This is a wrapper around gnuplot that intercepts allows the user to input
// SimulationConfig files and will compute MTF curves.
//
// For a simulation file, you can give the following options in the plot command
// in addition to previous gnuplot options:
//
// "simulation_id"/"s"  Gives the simulation id to plot within the file
// "orientation"/"o"    The orientation of the aperture [degrees]
//
// Author: Philip Salvaggio

#include "mats.h"

#include <csignal>
#include <iostream>
#include <thread>
#include <queue>
#include <unistd.h>

#include <opencv/cv.h>
#include <opencv/highgui.h>

using namespace cv;
using namespace std;

static Gnuplot gp;

void Plot(const string& command);

int main() {
  // Determine whether input is redirected or from the keyboard
  bool from_keyboard = isatty(STDIN_FILENO);

  // Set default plot setup
  gp << "set xlabel \"Spatial Frequency [cyc / pixel]\"\n"
     << "set ylabel \"MTF\"\n"
     << "set xrange [0:0.5]\n"
     << "set yrange [0:1]\n"
     << "set style fill solid 0.5\n";

  // Parse each line, if it's a plot command, we need to intercept it,
  // otherwise, just pass it on to gnuplot.
  string line;
  if (from_keyboard) cout << "mtfplot> ";
  while (getline(cin, line)) {
    if (mats::starts_with(line, "plot")) {
      Plot(line);
    } else if (line == "exit") {
      break;
    } else {
      gp << line << "\n" << endl;
    }
    if (from_keyboard) cout << "mtfplot> ";
  }
  gp << endl;

  return 0;
}


bool PlotConfigFile(const std::string& file,
                    list<string>& args,
                    vector<string>* plot_cmds);

bool PlotDataFile(const string& file,
                  list<string>& args,
                  vector<string>* plot_cmds);


void Plot(const string& command) {
  vector<string> plot_clauses;
  mats::explode(command.substr(5), ',', &plot_clauses);

  vector<string> plot_cmds;
  for (auto& plot_clause : plot_clauses) {
    vector<string> tokens;
    mats::explode(mats::trim(plot_clause), "[\\s]+", &tokens);

    if (!(mats::starts_with(tokens[0], "\"") &&
          mats::ends_with(tokens[0], "\""))) {
      cerr << "Error: Data files must be in qoutes." << endl;
      plot_cmds.clear();
      break;
    }

    string data_file = mats::ResolvePath(
        tokens[0].substr(1, tokens[0].size() - 2));
    if (!mats::file_exists(data_file)) {
      cerr << "Error: \"" << data_file << "\" does not exist." << endl;
      plot_cmds.clear();
      break;
    }

    list<string> args(begin(tokens) + 1, end(tokens));

    if (!PlotConfigFile(data_file, args, &plot_cmds)) {
      PlotDataFile(data_file, args, &plot_cmds);
    }
  }

  if (plot_cmds.empty()) return;

  gp << "plot ";
  for (size_t i = 0; i < plot_cmds.size(); i++) {
    if (i != 0) gp << ",";
    gp << plot_cmds[i];
  }

  gp << "\n" << endl;
}

void GetMtfProfile(const mats::SimulationConfig& sim_config,
                   int sim_index,
                   const mats::DetectorParameters& det_params,
                   const vector<double>& wavelengths,
                   const vector<double>& spectral_weighting,
                   double orientation,
                   string* plot_cmd) {
  // Create the telescope.
  mats::Telescope telescope(sim_config, sim_index, det_params);
  telescope.detector()->set_rows(512);
  telescope.detector()->set_cols(512);
  telescope.set_parallelism(false);
  telescope.set_include_detector_footprint(true);

  // Compute the theoretical 2D OTF of the telescope.
  Mat theoretical_otf;
  telescope.EffectiveOtf(wavelengths, spectral_weighting, 0, 0,
                         &theoretical_otf);
  Mat theoretical_2d_mtf = magnitude(theoretical_otf);

  vector<double> mtf_vals;
  GetRadialProfile(FFTShift(theoretical_2d_mtf), 
                   orientation * M_PI / 180,
                   &mtf_vals);

  vector<pair<double, double>> mtf;
  for (size_t i = 0; i < mtf_vals.size(); i++) {
    mtf.emplace_back(i / (2. * (mtf_vals.size() - 1)), mtf_vals[i]);
  }

  string title;
  if (sim_config.simulation(sim_index).has_name()) {
    title = sim_config.simulation(sim_index).name();
  } else {
    title = mats::StringPrintf("Simulation %d",
                sim_config.simulation(sim_index).simulation_id());
  }

  // Plot the MTfs
  stringstream ss;
  ss << gp.file1d(mtf) << "w l ";
  *plot_cmd = ss.str();
}

bool PlotConfigFile(const std::string& file,
                    list<string>& args,
                    vector<string>* plot_cmds) {
  // Initialize the model
  mats::SimulationConfig sim_config;
  mats::DetectorParameters det_params;
  if (!mats::MatsInit(file, &sim_config, &det_params, nullptr, nullptr)) {
    return false;
  }

  // Set up the spectral resolution of the simulation
  vector<vector<double>> raw_weighting;
  mats_io::TextFileReader::Parse(
      sim_config.spectral_weighting_filename(),
      &raw_weighting);
  const vector<double>& wavelengths(raw_weighting[0]);
  const vector<double>& spectral_weighting(raw_weighting[1]);

  // Parse arguments
  int sim_id = -1, sim_index = -1;
  double orientation = 0;
  for (auto it = begin(args); it != prev(end(args)) && it != end(args); ++it) {
    if (*it == "sid" || *it == "simulation_id") {
      sim_id = atoi(next(it)->c_str());
      it = args.erase(it);
      it = args.erase(it);
    }
    if (*it == "o" || *it == "orientation") {
      orientation = atof(next(it)->c_str());
      it = args.erase(it);
      it = args.erase(it);
    }
  }

  // Locate the simulation ID
  for (int i = 0; i < sim_config.simulation_size(); i++) {
    if (sim_config.simulation(i).simulation_id() == sim_id) {
      sim_index = i;
      break;
    }
  }


  int lower = (sim_index == -1) ? 0 : sim_index;
  int upper = (sim_index == -1) ? sim_config.simulation_size() : sim_index + 1;
  for (int i = lower; i < upper; i++) {
    string plot_cmd;
    GetMtfProfile(sim_config, i, det_params,
                  wavelengths, spectral_weighting, orientation,
                  &plot_cmd);

    string title = mats::StringPrintf("\"%s\" Simulation %d",
        mats::Basename(file).c_str(), i);
    if (sim_config.simulation(i).has_name()) {
      title = sim_config.simulation(i).name();
    }

    stringstream ss;
    ss << plot_cmd << " t \"" << title << "\"";
    for (const auto& token : args) {
      ss << token << " ";
    }

    plot_cmds->push_back(ss.str());
  }

  return true;
}

bool PlotDataFile(const string& file,
                  list<string>& args,
                  vector<string>* plot_cmds) {
  vector<vector<double>> raw_mtf_data;
  if (!mats_io::TextFileReader::Parse(file, &raw_mtf_data)) return false;

  vector<pair<double, double>> mtf_data;
  for (size_t i = 0; i < raw_mtf_data[0].size(); i++) {
    mtf_data.emplace_back(raw_mtf_data[0][i], raw_mtf_data[1][i]);
  }

  if (raw_mtf_data.size() >= 5) {
    vector<tuple<double, double, double>> mtf_bounds;
    for (size_t i = 0; i < raw_mtf_data[0].size(); i++) {
      mtf_bounds.emplace_back(raw_mtf_data[0][i], raw_mtf_data[3][i],
                              raw_mtf_data[4][i]);
    }

    stringstream ss;
    ss << gp.file1d(mtf_bounds) << "u 1:2:3 w filledcurves notitle";
    plot_cmds->push_back(ss.str());
  }

  stringstream ss;
  ss << gp.file1d(mtf_data) << "w l ";
  for (const auto& token : args) {
    ss << token << " ";
  }
  plot_cmds->push_back(ss.str());

  return true;
}
