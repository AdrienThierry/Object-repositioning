#include <stdio.h>
#include <vector>
#include "Superpixel.hpp"
#include "Point.hpp"

std::vector<struct Superpixel>* computeSuperpixels(cv::Mat img) {
	std::vector<struct Superpixel>* result = new std::vector<struct Superpixel>;

	// TODO : implement real function
	// TEST : center = pixel, 1 superpixel = 1 pixel = center
	for (int i = 0 ; i < img.rows ; i++) {
		for (int j = 0 ; j < img.cols ; j++) {
			// Get current pixel color
			int r, g, b;
			r = img.data[img.step[0]*i + img.step[1]*j + 2];
			g = img.data[img.step[0]*i + img.step[1]*j + 1];
			b = img.data[img.step[0]*i + img.step[1]*j + 0];

			// TODO : find center of superpixel
			
			Superpixel superpixel;
			Point pixel;

			pixel.x = j;
			pixel.y = i;

			superpixel.pixels.push_back(pixel);

			superpixel.color.r = r;
			superpixel.color.g = g;
			superpixel.color.b = b;

			result->push_back(superpixel);
		}
	}

	return result;
}

cv::Mat convertSuperpixelsToCV_Mat(std::vector<struct Superpixel>* superpixels, int rows, int cols) {
	cv::Mat result(rows, cols, CV_8UC3);

	// Create OpenCV matrix
	for (int i = 0 ; i < superpixels->size() ; i++) {
		for (int j = 0 ; j < superpixels->at(i).pixels.size() ; j++) {
			int row = superpixels->at(i).pixels.at(j).y;
			int col = superpixels->at(i).pixels.at(j).x;
			result.data[result.step[0]*row + result.step[1]*col + 0] = superpixels->at(i).color.b;
			result.data[result.step[0]*row + result.step[1]*col + 1] = superpixels->at(i).color.g;
			result.data[result.step[0]*row + result.step[1]*col + 2] = superpixels->at(i).color.r;
		}
	}
	
	return result;
}	


