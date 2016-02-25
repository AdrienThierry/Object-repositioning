#ifndef GMM_H_   /* Include guard */
#define GMM_H_

#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp>

#include "BoundingBox.hpp"

#define NB_GMM_CLUSTERS 5

struct GMM {
	cv::EM model;
	cv::Mat probs;
	cv::Mat labels;
	double weights[NB_GMM_CLUSTERS];
};

struct GMM computeGMMBackground(IplImage *imageIpl);
struct GMM computeGMMForeground(IplImage *imageIpl, struct BoundingBox bb);
void convertGMMLabelsToCV_Mat(IplImage** result, struct GMM *GMM, int rows, int cols);

#endif
