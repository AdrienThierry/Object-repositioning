#ifndef SUPERPIXEL_H_   /* Include guard */
#define SUPERPIXEL_H_

#include <opencv2/opencv.hpp>
#include <vector>
#include "Point.hpp"
#include "Color.hpp"

struct Superpixel {
	struct Point center; // Center of the superpixel
	std::vector<struct Point> pixels; // Pixels that belong to the superpixel
	struct Color color;
};

// Take as input an OpenCV matrix and returns a array of superpixels that compose the matrix
std::vector<struct Superpixel>* computeSuperpixels(cv::Mat img);

// Convert superpixels to an OpenCV matrix to be shown for debug purposes. Also takes as inputs
// the numbers of rows and cols in the original image
cv::Mat convertSuperpixelsToCV_Mat(std::vector<struct Superpixel>* superpixels, int rows, int cols);

#endif
