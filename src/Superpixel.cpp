#include <stdio.h>
#include <vector>
#include <unistd.h>
#include "Superpixel.hpp"
#include "Point.hpp"

#define CENTROID_RECT_SIZE 5 // Size of centroid when shown on screen
#define CENTROID_RECT_ALPHA 255 // Transparency of centroid when shown on screen

#define dbg_print1(buffer, string, arg) write(1, buffer, sprintf(buffer, string, arg))

std::vector<struct Superpixel>* computeSuperpixels(int** ilabels, cv::Mat img, int rows, int cols) {
	std::vector<struct Superpixel>* result = new std::vector<struct Superpixel>;

	// Look-up table to quickly find a superpixel by its label. The index in this array is the label.
	// The corresponding element is a pointer to the corresponding superpixel.
	std::vector<struct Superpixel*> superpixelByLabel(rows*cols);

	for (unsigned int i = 0 ; i < superpixelByLabel.size() ; i++) {
		superpixelByLabel.at(i) = NULL;
	}

	for (int i = 0 ; i < rows ; i++) {
		for (int j = 0 ; j < cols ; j++) {
			// Get current pixel label
			int label = ilabels[i][j];

			// If no superpixel exists for given label, create a new one
			if (superpixelByLabel.at(label) == NULL) {
				Superpixel *newSuperpixel = new Superpixel;
				newSuperpixel->color.r = 0;
				newSuperpixel->color.g = 0;
				newSuperpixel->color.b = 0;
				superpixelByLabel.at(label) = newSuperpixel;
			}			

			struct Point pixel;

			pixel.x = j;
			pixel.y = i;

			superpixelByLabel.at(label)->pixels.push_back(pixel);
		}
	}

	// Create Superpixel vector from superpixelByLabel
	for (unsigned int i = 0 ; i < superpixelByLabel.size() ; i++) {
		if (superpixelByLabel.at(i) != NULL) {
			result->push_back(*superpixelByLabel.at(i));
		}
	}

	// Compute the centroid and the mean color of each superpixel
	for (unsigned int i = 0 ; i < result->size() ; i++) {
		result->at(i).center.x = 0;
		result->at(i).center.y = 0;
		result->at(i).color.r = 0;
		result->at(i).color.g = 0;
		result->at(i).color.b = 0;
		for (unsigned int j = 0 ; j < result->at(i).pixels.size() ; j++) {
			result->at(i).center.x += result->at(i).pixels.at(j).x;
			result->at(i).center.y += result->at(i).pixels.at(j).y;
			result->at(i).color.r += img.data[img.step[0]*(result->at(i).pixels.at(j).y) + img.step[1]*(result->at(i).pixels.at(j).x) + 2];
			result->at(i).color.g += img.data[img.step[0]*(result->at(i).pixels.at(j).y) + img.step[1]*(result->at(i).pixels.at(j).x) + 1];
			result->at(i).color.b += img.data[img.step[0]*(result->at(i).pixels.at(j).y) + img.step[1]*(result->at(i).pixels.at(j).x) + 0];
		}
		result->at(i).center.x /= result->at(i).pixels.size();
		result->at(i).center.y /= result->at(i).pixels.size();
		result->at(i).color.r /= result->at(i).pixels.size();
		result->at(i).color.g /= result->at(i).pixels.size();
		result->at(i).color.b /= result->at(i).pixels.size();
	}

	return result;
}

void convertSuperpixelsToCV_Mat(IplImage **result, std::vector<struct Superpixel>* superpixels, int rows, int cols) {
	cv::Mat tmpResult = cv::Mat::zeros(rows, cols, CV_8UC3);

	// Create OpenCV matrix
	for (unsigned int i = 0 ; i < superpixels->size() ; i++) {
		for (unsigned int j = 0 ; j < superpixels->at(i).pixels.size() ; j++) {
			int row = superpixels->at(i).pixels.at(j).y;
			int col = superpixels->at(i).pixels.at(j).x;
			tmpResult.data[tmpResult.step[0]*row + tmpResult.step[1]*col + 0] = superpixels->at(i).color.b;
			tmpResult.data[tmpResult.step[0]*row + tmpResult.step[1]*col + 1] = superpixels->at(i).color.g;
			tmpResult.data[tmpResult.step[0]*row + tmpResult.step[1]*col + 2] = superpixels->at(i).color.r;
		}
	}

	cvReleaseImage(result);
	*result = cvCreateImage(cvSize(cols,rows),8,3);
	IplImage ipltemp = tmpResult;
	cvCopy(&ipltemp,*result);
}	

void convertSuperpixelsIntersectionToCV_Mat(IplImage **result, std::vector<struct Superpixel>* superpixels, int rows, int cols) {
	cv::Mat tmpResult = cv::Mat::zeros(rows, cols, CV_8UC3);

	int color; // How much white (between 0 and 255)

	// Create OpenCV matrix
	for (unsigned int i = 0 ; i < superpixels->size() ; i++) {
		if (superpixels->at(i).intersection == OUTSIDE)
			color = 255;
		else if (superpixels->at(i).intersection == INSIDE)
			color = 90;
		else
			color = 0;


		for (unsigned int j = 0 ; j < superpixels->at(i).pixels.size() ; j++) {
			int row = superpixels->at(i).pixels.at(j).y;
			int col = superpixels->at(i).pixels.at(j).x;
			tmpResult.data[tmpResult.step[0]*row + tmpResult.step[1]*col + 0] = color;
			tmpResult.data[tmpResult.step[0]*row + tmpResult.step[1]*col + 1] = color;
			tmpResult.data[tmpResult.step[0]*row + tmpResult.step[1]*col + 2] = color;
		}
	}

	cvReleaseImage(result);
	*result = cvCreateImage(cvSize(cols,rows),8,3);
	IplImage ipltemp = tmpResult;
	cvCopy(&ipltemp,*result);
}

void convertSaliencyToCV_Mat(IplImage** result, std::vector<struct Superpixel>* superpixels, int rows, int cols) {
	cv::Mat tmpResult = cv::Mat::zeros(rows, cols, CV_8UC3);

	// Find max of saliency for normalization
	float maxSaliency = 0.0;
	for (unsigned int i = 0 ; i < superpixels->size() ; i++) {
		if (superpixels->at(i).saliency > maxSaliency)
			maxSaliency = superpixels->at(i).saliency;
	}

	int color; // How much white (between 0 and 255)

	// Create OpenCV matrix
	for (unsigned int i = 0 ; i < superpixels->size() ; i++) {
		color = (int)(255.0 * superpixels->at(i).saliency / maxSaliency);


		for (unsigned int j = 0 ; j < superpixels->at(i).pixels.size() ; j++) {
			int row = superpixels->at(i).pixels.at(j).y;
			int col = superpixels->at(i).pixels.at(j).x;
			tmpResult.data[tmpResult.step[0]*row + tmpResult.step[1]*col + 0] = color;
			tmpResult.data[tmpResult.step[0]*row + tmpResult.step[1]*col + 1] = color;
			tmpResult.data[tmpResult.step[0]*row + tmpResult.step[1]*col + 2] = color;
		}
	}

	cvReleaseImage(result);
	*result = cvCreateImage(cvSize(cols,rows),8,3);
	IplImage ipltemp = tmpResult;
	cvCopy(&ipltemp,*result);

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

std::vector<float> getSaliencyBackgroundLUT(std::vector<struct Superpixel>* superpixels, int rows, int cols) {
	std::vector<float> result(rows*cols);

	for (int i = 0 ; i < rows*cols ; i++) {
		result.at(i) = 0.0;
	}

	for (unsigned int i = 0 ; i < superpixels->size() ; i++) {
		for (unsigned int j = 0 ; j < superpixels->at(i).pixels.size() ; j++) {
			int row = superpixels->at(i).pixels.at(j).y;
			int col = superpixels->at(i).pixels.at(j).x;
			result.at(row*cols+col) = 1.0 - superpixels->at(i).saliency; // WARNING : 1 - saliency for background (see article)
		}
	}

	return result;

}

std::vector<float> getSaliencyForegroundLUT(std::vector<struct Superpixel>* superpixels, struct BoundingBox bb) {
	// Determine min x, max x, min y and max y in bounding box
	computeBBEnds(&bb);
	int minX = bb.ends[0];
	int maxX = bb.ends[1];
	int minY = bb.ends[2];
	int maxY = bb.ends[3];

	int rows = maxY - minY + 1;
	int cols = maxX - minX + 1;

	std::vector<float> result(rows*cols);

	for (int i = 0 ; i < rows*cols ; i++) {
		result.at(i) = 0.0;
	}

	for (unsigned int i = 0 ; i < superpixels->size() ; i++) {
		for (unsigned int j = 0 ; j < superpixels->at(i).pixels.size() ; j++) {
			int row = superpixels->at(i).pixels.at(j).y;
			int col = superpixels->at(i).pixels.at(j).x;
			if (row >= minY && row <= maxY && col >= minX && col <= maxX)
				result.at((row-minY)*cols+(col-minX)) = superpixels->at(i).saliency;
		}
	}

	return result;
}

