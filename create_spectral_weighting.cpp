// File Description
// Author: Philip Salvaggio

#include "mats.h"

#include "io/detector_reader.h"

#include <iostream>
#include <numeric>
#include <vector>

using namespace std;

const static double kMeters = 1;
const static double kMicrons = 1e-6;
const static double kNanometers = 1e-9;
const static double kDetector = 10;
const static double kWavelengthRange = 100;

int main(int argc, char** argv) {
  if (argc < 2) {
    cerr << "Usage: " << argv[0] << " (-(w|m|um|nm|d) arg)+" << endl
         << endl
         << "Arguments:" << endl
         << "-w    Wavelength domain. Format: \"start:increment:end\" [m]" 
         << endl
         << "-m    Text file with wavelengths in meters." << endl
         << "-um   Text file with wavelengths in microns." << endl
         << "-nm   Text file with wavelengths in nanometers." << endl
         << "-d    Detector config file, uses QE spectrum." << endl;
    return 1;
  }

  double units = kMeters;

  vector<double> wavelengths;
  vector<double> spectra;
  
  for (int i = 1; i < argc; i++) {
    string param = argv[i];
    if (param == "-m") {
      units = kMeters;
    } else if (param == "-um") {
      units = kMicrons;
    } else if (param == "-nm") {
      units = kNanometers;
    } else if (param == "-d") {
      units = kDetector;
    } else if (param == "-w") {
      units = kWavelengthRange;
    } else if (mats::file_exists(param)) {
      vector<vector<double>> data;

      // If this was a detector file, read in the config file.
      mats::DetectorParameters det_params;
      if (units == kDetector) {
        mats_io::DetectorReader det_reader;
        if (!det_reader.Read(param, &det_params)) {
          cerr << "Could not read detector file." << endl;
          return 1;
        }
      }

      // If this is the first data source, then we need to initialize th
      // wavelength domain.
      if (wavelengths.empty()) {
        if (units == kDetector) {  // Detector -> use QE spectrum domain
          for (int j = 0; j < det_params.band(0).wavelength_size(); j++) {
            wavelengths.push_back(det_params.band(0).wavelength(j));
            spectra.push_back(det_params.band(0).quantum_efficiency(j));
          }
        } else {  // Data file -> use its domain in meters
          mats_io::TextFileReader::Parse(param, &data);
          if (data.size() < 2) {
            cerr << "First data file not properly formatted." << endl;
            return 1;
          }
          for (size_t j = 0; j < data[0].size(); j++) {
            wavelengths.push_back(data[0][j] * units);
            spectra.push_back(data[1][j]);
          }
        }

      // Otherwise, resample to the wavelength domain and multiply
      } else {
        if (units == kDetector) {
          mats::SimulationConfig sim_config;
          mats::Detector det(det_params);
          
          vector<double> qe;
          det.GetQESpectrum(wavelengths, 0, &qe);
          for (size_t j = 0; j < qe.size(); j++) {
            spectra[j] *= qe[j];
          }
        } else {
          vector<double> tmp_wavelengths(wavelengths);
          for (auto& tmp : tmp_wavelengths) tmp /= units;

          cerr << "Resampling " << param << endl;
          mats_io::TextFileReader::Resample(param, tmp_wavelengths, &data);
          if (data.size() < 1) {
            cerr << "No data in " << param << endl;
            return 1;
          }
          if (spectra.size() != data[0].size()) {
            cerr << "Size mismatch: " << spectra.size() << " vs. " 
                 << data[0].size() << endl;
            return 1;
          }
          for (size_t j = 0; j < data[0].size(); j++) {
            spectra[j] *= data[0][j];
          }
        }
      }
      units = kMeters;
    } else if (units == kWavelengthRange) {
      vector<string> parts;
      mats::explode(param, ':', &parts);
      if (parts.size() != 3) {
        cerr << "Wavelength format start:increment:end [m]" << endl;
        return 1;
      }
      if (wavelengths.empty()) {
        double start = atof(parts[0].c_str());
        double increment = atof(parts[1].c_str());
        double end = atof(parts[2].c_str());
        for (double lambda = start; lambda <= end; lambda += increment) {
          wavelengths.push_back(lambda);
          spectra.push_back(1);
        }
      }
      units = kMeters;
    }
  }

  double total = accumulate(begin(spectra), end(spectra), 0.);
  for (auto& tmp : spectra) tmp /= total;

  for (size_t i = 0; i < wavelengths.size(); i++) {
    cout << wavelengths[i] << "\t" << spectra[i] << endl;
  }

  return 0;
}
