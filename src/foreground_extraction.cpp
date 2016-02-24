#include "foreground_extraction.hpp"

#define NB_GMM 3

cv::Mat* computeProbsBackground(cv::Mat *image) {
	cv::EM model(NB_GMM);
	cv::Mat samples;
	cv::Mat logLikelihood;
	cv::Mat labels;
	cv::Mat probs;

	image->reshape(1,image->rows*image->cols).convertTo(samples,CV_64FC1,1.0/255.0);
	model.train(samples, logLikelihood, labels, probs);

	probs = probs.reshape(NB_GMM, image->rows);
	labels = labels.reshape(1, image->rows);

	// Only return probabilities of background (first GMM cluster)
	cv::Mat *result = new cv::Mat(image->rows, image->cols, CV_64FC1);

	for (int i = 0 ; i < image->rows ; i++) {
		for (int j = 0 ; j < image->cols ; j++) {
			if (labels.data[labels.step[0]*i + labels.step[1]*j] == 0) {
				result->data[result->step[0]*i + result->step[1]*j] = probs.data[probs.step[0]*i + probs.step[1]*j + 0];
			}
			else {
				result->data[result->step[0]*i + result->step[1]*j] = 0;
			}
		}
	}

	return result;
}

cv::Mat* convertProbsToCV_Mat(cv::Mat *probs) {
	cv::Mat *result = new cv::Mat(probs->rows, probs->cols, CV_8UC3);
	for (int i = 0 ; i < probs->rows ; i++) {
		for (int j = 0 ; j < probs->cols ; j++) {
			result->data[result->step[0]*i + result->step[1]*j + 0] = probs->data[probs->step[0]*i + probs->step[1]*j] * 255.0;
			result->data[result->step[0]*i + result->step[1]*j + 1] = probs->data[probs->step[0]*i + probs->step[1]*j] * 255.0;
			result->data[result->step[0]*i + result->step[1]*j + 2] = probs->data[probs->step[0]*i + probs->step[1]*j] * 255.0;
		}
	}

	return result;
}
