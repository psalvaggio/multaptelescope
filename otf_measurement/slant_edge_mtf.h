// File Description
// Author: Philip Salvaggio

#ifndef SLANT_EDGE_MTF_H
#define SLANT_EDGE_MTF_H

#include <vector>
#include <opencv/cv.h>

class SlantEdgeMtf {
 public:
  SlantEdgeMtf();
  ~SlantEdgeMtf();

  // Analyze an image patch with a slant edge.
  //
  // Arguments:
  //   image        The input image. It should have a slanted edge that runs the
  //                whole way through the image. That is, the edge should 
  //                appear in every row/column of the image.
  //   orientation  Output: The detected orientation of the edge. The
  //                frequencies in the 2D MTF will be orthogonal to this
  //                orientation. [radians CCW of x-axis]
  //   mtf          Output: Contains the 1D MTF data in cycles/pixel. The
  //                frequency at element i in the list is given by
  //                i / mtf.size() cycles/pixel.
  void Analyze(const cv::Mat& image,
               double* orientation,
               std::vector<double>* mtf);

  // Overlays the given line in green on the given grayscale image.
  //
  // Arguments:
  //  image  The image onto which to overlay the line.
  //  line   Normal for of the line [a, b, c] (a*x + b*y = c)
  //
  // Returns:
  //  An RGB image with the line overlaid in green.
  cv::Mat OverlayLine(const cv::Mat& image, const double* line);

 private:
  // Detects the edge in the image.
  //
  // Arguments:
  //  image  The input image. The slanted edge should be present in every
  //         row/column of the image.
  //  edge   Returns the normal form of the equation of the edge. Should be a
  //         pointer to a 3-element double array.
  //
  // Returns:
  //  Whether or not the edge was successfully detected.
  bool DetectEdge(const cv::Mat& image, double* edge);

  // Detects the resolution that can be supported with the detected line
  // orientation.
  //
  // Arguments:
  //  image  The image being examined. Needed to get the image size.
  //  edge   The edge detected in the image.
  int GetSamplesPerPixel(const cv::Mat& image, const double* edge);

  // Generate the edge spread function from the image data.
  //
  // Arguments:
  //  image             The image being examined.
  //  edge              The edge detected in the image.
  //  samples_per_pixel The number of samples per pixel.`
  //  esf               Output: The edge spread function.
  void GenerateEsf(const cv::Mat& image,
                   const double* edge,
                   int samples_per_pixel,
                   std::vector<double>* esf);

  // Smooth the ESF with a 3-element uniform blur kernel.
  //
  // Arguments:
  //  esf  Input/Otput: The edge spread function.
  void SmoothEsf(std::vector<double>* esf);
};

#endif  // SLANT_EDGE_MTF_H
