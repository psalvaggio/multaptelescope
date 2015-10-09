// This is a common include file for the main components of the MATS model.
// Author: Philip Salvaggio

#ifndef MATS_H
#define MATS_H

// Core headers of the library.
#include "base/assertions.h"
#include "base/aperture_parameters.pb.h"
#include "base/detector.h"
#include "base/detector_parameters.pb.h"
#include "base/filesystem.h"
#include "base/linear_interpolator.h"
#include "base/math_utils.h"
#include "base/mats_init.h"
#include "base/menu_application.h"
#include "base/opencv_utils.h"
#include "base/photon_noise.h"
#include "base/pupil_function.h"
#include "base/simulation_config.pb.h"
#include "base/str_utils.h"
#include "base/subprocess.h"
#include "base/telescope.h"
#include "base/wait_queue.h"

// Headers for image restoration.
#include "deconvolution/constrained_least_squares.h"

// Input/output headers.
#include "io/logging.h"
#include "io/envi_image_header.pb.h"
#include "io/envi_image_reader.h"
#include "io/envi_image_writer.h"
#include "io/sbig_detector.h"
#include "io/text_file_reader.h"

// Aperture module headers. A Registry pattern is used here, so only aperture.h
// should be needed for most applications. Others can be explicitly included
// when needed.
#include "optical_designs/aperture.h"

// OTF measurement headers for laboratory applications.
#include "otf_measurement/slant_edge_mtf.h"
#include "otf_measurement/usaf_1951_target.h"

// Aperture Optimization headers.
#include "aperture_optimization/golay_fitness_function.h"
#include "aperture_optimization/global_sparse_aperture.h"
#include "aperture_optimization/local_sparse_aperture.h"
#include "genetic/genetic_algorithm.h"

#endif  // MATS_H
