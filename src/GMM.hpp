#ifndef GMM_H_   /* Include guard */
#define GMM_H_

#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp>
#include <vector>

#include "BoundingBox.hpp"

#define NB_GMM_CLUSTERS 5

struct GMM {
	cv::EM model;
	std::vector<std::vector<double> > probs;
	std::vector<std::vector<int> > labels;
	std::vector<std::vector<double> > weightedLL; // Weighted minux log likelihoods
	double weights[NB_GMM_CLUSTERS]; // Weights of the GMM components
};

struct GMM computeGMM(IplImage *imageIpl);
void computeWeightedLogLikelihoods(struct GMM* gmm);
void convertGMMLabelsToCV_Mat(IplImage** result, struct GMM *GMM, int rows, int cols);
void convertGMMWeightedLLToCV_Mat(IplImage** result, struct GMM *GMM, int rows, int cols);

#endif
