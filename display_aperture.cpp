// Displays an aperture specified in a SimulationConfig protobuf.
// Author: Philip Salvaggio

#include "mats.h"

#ifdef __APPLE__
#include <GLUT/glut.h>
#elif __linux
#include <GL/glew.h>
#include <GL/freeglut.h>
#else
#error "Unsupported OS"
#endif

#include <iostream>
#include <opencv/highgui.h>
#include <gflags/gflags.h>


DEFINE_string(config_file, "", "Required: SimulationConfig file");
DEFINE_int32(simulation_id, 0, "Optional: simualtion ID for multi-simulation "
                               "config files");
DEFINE_int32(rows, 768, "Number of rows to display");
DEFINE_int32(cols, 1024, "Number of columns to display");
DEFINE_double(pixel_pitch, 36e-6, "Pixel pitch to define the scale of the "
                                  "display");

using namespace std;
using namespace cv;

static unique_ptr<Aperture> ap;

void draw() {
  Mat mask = ap->GetApertureMask();

  Mat roi;
  if (mask.rows > FLAGS_rows || mask.cols > FLAGS_cols) {
    int x0 = mask.cols / 2 - FLAGS_cols / 2;
    int y0 = mask.rows / 2 - FLAGS_rows / 2;
    roi = mask(Range(y0, y0 + FLAGS_rows), Range(x0, x0 + FLAGS_rows));
  } else {
    roi = mask;
  }

  imwrite("mask.png", ByteScale(mask));

  int y0 = FLAGS_rows / 2 - roi.rows / 2;
  int x0 = FLAGS_cols / 2 - roi.cols / 2;

  cout << "x0: " << x0 << ", y0: " << y0 << endl;

  
  glClearColor(0, 0, 0, 1);
  glClear(GL_COLOR_BUFFER_BIT);
  glFlush();

  glPushMatrix();
  glLoadIdentity();
  glBegin(GL_POINTS);
  for (int y = 0; y < FLAGS_rows; y++) {
    for (int x = 0; x < FLAGS_cols; x++) {
      double value = 0;
      if (x >= x0 && y >= y0 && y - y0 < roi.rows && x - x0 < roi.cols) {
        value = roi.at<double>(y - y0, x - x0);
      }
      glColor3d(value, value, value);
      glVertex2i(x, y);
    }
  }
  glEnd();
  glPopMatrix();
  glFlush();
  glutSwapBuffers();
}

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);

  mats::SimulationConfig sim_config;
  if (!mats::MatsInit(FLAGS_config_file,
                      &sim_config,
                      nullptr, nullptr, nullptr)) {
    cerr << "Could not read simulation file." << endl;
    return 1;
  }

  int sim_index = 0;
  for (int i = 0; i < sim_config.simulation_size(); i++) {
    if (sim_config.simulation(i).simulation_id() == FLAGS_simulation_id) {
      sim_index = i;
      break;
    }
  }

  ap.reset(ApertureFactory::Create(sim_config.simulation(sim_index)));
  if (ap.get() == nullptr) {
    cerr << "Invalid aperture configuration" << endl;
    return 1;
  }

  int aperture_size = round(ap->encircled_diameter() / FLAGS_pixel_pitch);
  cout << "Aperture Size: " << aperture_size << endl;

  if (aperture_size == 0) {
    cerr << "Aperture is too small to display." << endl;
    return 1;
  }

  sim_config.set_array_size(aperture_size);
  ap.reset(ApertureFactory::Create(sim_config.simulation(sim_index)));

  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
  glutInitWindowSize(FLAGS_cols, FLAGS_rows);
  glutCreateWindow("Aperuture");
  glutFullScreen();
  glutDisplayFunc(draw);

  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  glOrtho(0, FLAGS_cols, FLAGS_rows, 0, 1, -1);
  glViewport(0, 0, FLAGS_cols, FLAGS_rows);
  glMatrixMode(GL_MODELVIEW);

  glutMainLoop();

  return 0;
}
