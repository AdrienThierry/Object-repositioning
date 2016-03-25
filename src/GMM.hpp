#ifndef GMM_H_   /* Include guard */
#define GMM_H_

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <unistd.h>
#include <semaphore.h>
#include <vector>

#include "BoundingBox.hpp"

#define NB_GMM_CLUSTERS 5

struct GMM {
	cv::EM model;
	std::vector<std::vector<int> > labels;
	std::vector<std::vector<double> > weightedProbs;
	std::vector<std::vector<double> > weightedLL; // Weighted minus log likelihoods
	double weights[NB_GMM_CLUSTERS]; // Weights of the GMM components
};

struct GMM_arg_struct {
	struct GMM *result;
	IplImage *imageIpl;
	std::vector<float>* saliencyLUT;
	sem_t *semaphore; // Rdv with main
};

void *computeGMM(void *args);

IplImage preProcessingForegroundGMM(IplImage *image, BoundingBox bb); // Creates an image that only contains the pixels inside the bounding box

// Completes GMM model with black pixels around bounding box. "rows" and "cols" are RESULTING size
void postProcessingForegroundGMM(struct GMM *GMM, BoundingBox bb, int rows, int cols);

void convertGMMLabelsToCV_Mat(IplImage** result, struct GMM *GMM, int rows, int cols);
void convertGMMWeightedLLToCV_Mat(IplImage** result, struct GMM *GMM, int rows, int cols);
void convertGMMWeightedProbsToCV_Mat(IplImage** result, struct GMM *GMM, int rows, int cols);

#endif
