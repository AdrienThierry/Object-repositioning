#ifndef SUPERPIXEL_H_   /* Include guard */
#define SUPERPIXEL_H_

#include <opencv2/opencv.hpp>
#include <vector>
#include <SDL2/SDL.h>
#include "Point.hpp"
#include "Color.hpp"

typedef enum {INTERSECT, INSIDE, OUTSIDE} IntersectionWithBB;

struct Superpixel {
	struct Point center; // Center of the superpixel
	std::vector<struct Point> pixels; // Pixels that belong to the superpixel
	struct Color color;
	IntersectionWithBB intersection;
	float saliency;
};

// Take as input a label matrix and returns a array of superpixels that compose the matrix
std::vector<struct Superpixel>* computeSuperpixels(int** ilabels, cv::Mat img, int rows, int cols);

// Convert superpixels to an OpenCV matrix
cv::Mat convertSuperpixelsToCV_Mat(std::vector<struct Superpixel>* superpixels, int rows, int cols);

// Convert superpixels intersections to an OpenCV matrix
// Superpixels entirely outside BB are shown in WHITE
// Superpixels entirely inside BB are shown in GREY
// Superpixels that cross BB are shown in BLACK
cv::Mat convertSuperpixelsIntersectionToCV_Mat(std::vector<struct Superpixel>* superpixels, int rows, int cols);

// Convert superpixels saliency to an OpenCV matrix
cv::Mat convertSaliencyToCV_Mat(std::vector<struct Superpixel>* superpixels, int rows, int cols);

// Show centroids of superpixels with an SDL_Renderer
void showCentroids(SDL_Renderer *ren, std::vector<struct Superpixel>* superpixels);

#endif
