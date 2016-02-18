#include <stdio.h>
#include <vector>
#include <unistd.h>
#include "Superpixel.hpp"
#include "Point.hpp"

#define CENTROID_RECT_SIZE 5 // Size of centroid when shown on screen
#define CENTROID_RECT_ALPHA 255 // Transparency of centroid when shown on screen

#define dbg_print1(buffer, string, arg) write(1, buffer, sprintf(buffer, string, arg))

std::vector<struct Superpixel>* computeSuperpixels(cv::Mat img) {
	std::vector<struct Superpixel>* result = new std::vector<struct Superpixel>;

	// Look-up table to quickly find a superpixel by its color. The index in this array is the color.
	// The corresponding element is a pointer to the corresponding superpixel.
	std::vector<struct Superpixel*> superpixelByColor(256*256*256);

	for (unsigned int i = 0 ; i < superpixelByColor.size() ; i++) {
		superpixelByColor.at(i) = NULL;
	}

	for (int i = 0 ; i < img.rows ; i++) {
		for (int j = 0 ; j < img.cols ; j++) {
			// Get current pixel color
			int r, g, b;
			r = img.data[img.step[0]*i + img.step[1]*j + 2];
			g = img.data[img.step[0]*i + img.step[1]*j + 1];
			b = img.data[img.step[0]*i + img.step[1]*j + 0];
			int colorIndex = (r << 16) + (g << 8) + b;

			// If no superpixel exists for given color, create a new one
			if (superpixelByColor.at(colorIndex) == NULL) {
				Superpixel *newSuperpixel = new Superpixel;
				newSuperpixel->color.r = r;
				newSuperpixel->color.g = g;
				newSuperpixel->color.b = b;
				superpixelByColor.at(colorIndex) = newSuperpixel;
			}			

			struct Point pixel;

			pixel.x = j;
			pixel.y = i;

			superpixelByColor.at(colorIndex)->pixels.push_back(pixel);
		}
	}

	// Create Superpixel vector from superpixelByColor
	for (unsigned int i = 0 ; i < superpixelByColor.size() ; i++) {
		if (superpixelByColor.at(i) != NULL) {
			result->push_back(*superpixelByColor.at(i));
		}
	}

	// Compute the centroid of each superpixel
	for (unsigned int i = 0 ; i < result->size() ; i++) {
		result->at(i).center.x = 0;
		result->at(i).center.y = 0;
		for (unsigned int j = 0 ; j < result->at(i).pixels.size() ; j++) {
			result->at(i).center.x += result->at(i).pixels.at(j).x;
			result->at(i).center.y += result->at(i).pixels.at(j).y;
		}
		result->at(i).center.x /= result->at(i).pixels.size();
		result->at(i).center.y /= result->at(i).pixels.size();
	}

	return result;
}

cv::Mat convertSuperpixelsToCV_Mat(std::vector<struct Superpixel>* superpixels, int rows, int cols) {
	cv::Mat result = cv::Mat::zeros(rows, cols, CV_8UC3);

	// Create OpenCV matrix
	for (unsigned int i = 0 ; i < superpixels->size() ; i++) {
		for (unsigned int j = 0 ; j < superpixels->at(i).pixels.size() ; j++) {
			int row = superpixels->at(i).pixels.at(j).y;
			int col = superpixels->at(i).pixels.at(j).x;
			result.data[result.step[0]*row + result.step[1]*col + 0] = superpixels->at(i).color.b;
			result.data[result.step[0]*row + result.step[1]*col + 1] = superpixels->at(i).color.g;
			result.data[result.step[0]*row + result.step[1]*col + 2] = superpixels->at(i).color.r;
		}
	}
	
	return result;
}	

void showCentroids(SDL_Renderer *ren, std::vector<struct Superpixel>* superpixels) {
	SDL_Rect r;
    r.w = CENTROID_RECT_SIZE;
    r.h = CENTROID_RECT_SIZE;

	for (unsigned int i = 0 ; i < superpixels->size() ; i++) {
		SDL_SetRenderDrawColor( ren, superpixels->at(i).color.r / 2, superpixels->at(i).color.g / 2, superpixels->at(i).color.b / 2, CENTROID_RECT_ALPHA );

		r.x = superpixels->at(i).center.x;
		r.y = superpixels->at(i).center.y;

		// Render rect
		SDL_RenderFillRect( ren, &r );
	}
}

