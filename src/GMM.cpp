#include <stdio.h>
#include "GMM.hpp"

struct GMM computeGMMBackground(cv::Mat *image) {
	struct GMM result;	

	cv::EM model(NB_GMM_CLUSTERS);
	cv::Mat samples;
	cv::Mat logLikelihood;
	cv::Mat labels;
	cv::Mat probs;

	image->reshape(1,image->rows*image->cols).convertTo(samples,CV_64FC1,1.0/255.0);
	model.train(samples, logLikelihood, labels, probs);

	probs = probs.reshape(NB_GMM_CLUSTERS, image->rows);
	labels = labels.reshape(1, image->rows);

	result.model = model;
	result.probs = probs;
	result.labels = labels;

	for (int i = 0 ; i < NB_GMM_CLUSTERS ; i++) {
		cv::Mat weights = model.get<cv::Mat>("weights");
		result.weights[i] = weights.data[weights.step[0]*i + weights.step[1]*0];
		printf("%f\n", result.weights[i]);
	}

	return result;
}
