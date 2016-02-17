#ifndef SUPERPIXEL_H_   /* Include guard */
#define SUPERPIXEL_H_

#include <opencv2/opencv.hpp>
#include "Point.hpp"
#include "Color.hpp"

struct Superpixel {
	struct Point center; // Center of the superpixel
	struct Point* pixels; // Pixels that belong to the superpixel
	int nbPoints;
	struct Color color;
};

struct SuperpixelArray {
	struct Superpixel **superpixels;
	int length;
};

// Take as input an OpenCV matrix and returns a array of superpixels that compose the matrix
struct SuperpixelArray computeSuperpixels(cv::Mat img);

// Convert superpixels to an OpenCV matrix to be shown for debug purposes. Also takes as inputs
// the numbers of rows and cols in the original image
cv::Mat convertSuperpixelsToCV_Mat(SuperpixelArray superpixels, int rows, int cols);

#endif
