#include <stdio.h>
#include "GMM.hpp"

struct GMM computeGMMBackground(IplImage *imageIpl) {
	struct GMM result;	

	cv::Mat image(imageIpl, true);

	cv::EM model(NB_GMM_CLUSTERS);
	cv::Mat samples;
	cv::Mat logLikelihood;
	cv::Mat labels;
	cv::Mat probs;

	image.reshape(1,image.rows*image.cols).convertTo(samples,CV_64FC1,1.0/255.0);
	model.train(samples, logLikelihood, labels, probs);

	probs = probs.reshape(NB_GMM_CLUSTERS, image.rows);
	labels = labels.reshape(1, image.rows);

	result.model = model;
	result.probs = probs;
	result.labels = labels;

	for (int i = 0 ; i < NB_GMM_CLUSTERS ; i++) {
		cv::Mat weights = model.get<cv::Mat>("weights");
		result.weights[i] = weights.data[weights.step[0]*i + weights.step[1]*0];
	}

	return result;
}

void convertGMMLabelsToCV_Mat(IplImage** result, struct GMM *GMM, int rows, int cols) {
	cv::Mat tmpResult = cv::Mat::zeros(rows, cols, CV_8UC3);

	// Create OpenCV matrix
	for (int i = 0 ; i < tmpResult.rows ; i++) {
		for (int j = 0 ; j < tmpResult.cols ; j++) {
			int color = (int)(255.0 / (float)(GMM->labels.data[GMM->labels.step[0]*i + GMM->labels.step[1]*j]));

			tmpResult.data[tmpResult.step[0]*i + tmpResult.step[1]*j + 0] = color;
			tmpResult.data[tmpResult.step[0]*i + tmpResult.step[1]*j + 1] = color;
			tmpResult.data[tmpResult.step[0]*i + tmpResult.step[1]*j + 2] = color;
			
		}
	}

	cvReleaseImage(result);
	*result = cvCreateImage(cvSize(cols,rows),8,3);
	IplImage ipltemp = tmpResult;
	cvCopy(&ipltemp,*result);
}
