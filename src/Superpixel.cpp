#include <stdio.h>
#include "Superpixel.hpp"
#include "Point.hpp"

struct SuperpixelArray computeSuperpixels(cv::Mat img) {
	// Allocates an array of superpixel pointers that has the same size as the input image
	SuperpixelArray result;
	result.superpixels = (Superpixel**)(malloc(img.cols*img.rows*sizeof(struct Superpixel*)));
	result.length = img.cols*img.rows;
	
	for (int i = 0 ; i < img.cols*img.rows ; i ++) {
		result.superpixels[i] = (Superpixel*)(malloc(sizeof(struct Superpixel)));
		result.superpixels[i]->pixels = (Point*)(malloc(img.cols*img.rows*sizeof(struct Point)));
		result.superpixels[i]->nbPoints = 0;
	}

	// TODO : implement real function
	// TEST : center = pixel, 1 superpixel = 1 pixel = center
	for (int i = 0 ; i < img.rows ; i++) {
		for (int j = 0 ; j < img.cols ; j++) {
			result.superpixels[i * img.cols + j]->center.x = j;
			result.superpixels[i * img.cols + j]->center.y = i;
			
			result.superpixels[i * img.cols + j]->pixels[0].x = j;
			result.superpixels[i * img.cols + j]->pixels[0].y = i;

			result.superpixels[i * img.cols + j]->color.r = img.data[img.step[0]*i + img.step[1]*j + 2];
			result.superpixels[i * img.cols + j]->color.g = img.data[img.step[0]*i + img.step[1]*j + 1];;
			result.superpixels[i * img.cols + j]->color.b = img.data[img.step[0]*i + img.step[1]*j + 0];;

			result.superpixels[i * img.cols + j]->nbPoints++;
		}
	}

	return result;
}

cv::Mat convertSuperpixelsToCV_Mat(SuperpixelArray superpixels, int rows, int cols) {
	cv::Mat result(rows, cols, CV_8UC3);

	for (int i = 0 ; i < superpixels.length ; i++) {
		for (int j = 0 ; j < superpixels.superpixels[i]->nbPoints ; j++) {
			int row = superpixels.superpixels[i]->pixels[j].y;
			int col = superpixels.superpixels[i]->pixels[j].x;
			result.data[result.step[0]*row + result.step[1]*col + 0] = superpixels.superpixels[i]->color.b;
			result.data[result.step[0]*row + result.step[1]*col + 1] = superpixels.superpixels[i]->color.g;
			result.data[result.step[0]*row + result.step[1]*col + 2] = superpixels.superpixels[i]->color.r;
		}
	}
	
	return result;
}	


