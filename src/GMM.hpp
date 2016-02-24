#ifndef GMM_H_   /* Include guard */
#define GMM_H_

#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp>
#include <vector>

#define NB_GMM_CLUSTERS 5

struct GMM {
	cv::EM model;
	cv::Mat probs;
	cv::Mat labels;
	vector<float> weights;
};

struct GMM computeGMMBackground(cv::Mat *image);
cv::Mat* convertGMMLabelsToCV_Mat(struct GMM *GMM);

#endif
