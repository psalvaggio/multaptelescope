// Function for the generation of a Hamming window.
// Author: Philip Salvaggio

#ifndef HAMMING_WINDOW_H
#define HAMMING_WINDOW_H

// Construct a Hamming window function.
//
// Arguments:
//   size    The size of the array.
//   center  The center index of the function.
//   hamming Output: The resulting Hamming window.
void HammingWindow(int size, int center, double* hamming);

#endif  // HAMMING_WINDOW_H
