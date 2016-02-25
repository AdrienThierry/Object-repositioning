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

struct GMM computeGMMForeground(IplImage *imageIpl, struct BoundingBox bb) {
	struct GMM result;	

	cv::Mat image(imageIpl, true);

	cv::EM model(NB_GMM_CLUSTERS);
	cv::Mat samples;
	cv::Mat logLikelihood;
	cv::Mat labels;
	cv::Mat probs;

	// Determine min x, max x, min y and max y in bounding box
	int minX = bb.points[0].x;
	int maxX = bb.points[0].x;
	int minY = bb.points[0].y;
	int maxY = bb.points[0].y;
	for (int i = 0 ; i < 4 ; i++) {
		if (bb.points[i].x < minX)
			minX = bb.points[i].x;

		if (bb.points[i].x > maxX)
			maxX = bb.points[i].x;

		if (bb.points[i].y < minY)
			minY = bb.points[i].y;
	
		if (bb.points[i].y > maxY)
			maxY = bb.points[i].y;
	}

	cv::Mat foreground = cv::Mat::zeros(maxY-minY+1, maxX-minX+1, CV_8UC3);
	for (int i = 0 ; i < foreground.rows ; i++) {
		for (int j = 0 ; j < foreground.cols ; j++) {
			foreground.data[foreground.step[0]*i + foreground.step[1]*j + 0] = image.data[image.step[0]*(i+minY) + image.step[1]*(j+minX) + 0];
			foreground.data[foreground.step[0]*i + foreground.step[1]*j + 1] = image.data[image.step[0]*(i+minY) + image.step[1]*(j+minX) + 1];
			foreground.data[foreground.step[0]*i + foreground.step[1]*j + 2] = image.data[image.step[0]*(i+minY) + image.step[1]*(j+minX) + 2];
		}
	}

	foreground.reshape(1,foreground.rows*foreground.cols).convertTo(samples,CV_64FC1,1.0/255.0);
	model.train(samples, logLikelihood, labels, probs);

	probs = probs.reshape(NB_GMM_CLUSTERS, foreground.rows);
	labels = labels.reshape(1, foreground.rows);

	result.model = model;
	result.probs = cv::Mat::zeros(image.rows, image.cols, CV_64FC1);
	result.labels = cv::Mat::zeros(image.rows, image.cols, CV_32SC1);
	for (int i = 0 ; i < foreground.rows ; i++) {
		for (int j = 0 ; j < foreground.cols ; j++) {
			result.probs.data[result.probs.step[0]*(i+minY) + result.probs.step[1]*(j+minX)] = probs.data[probs.step[0]*i + probs.step[1]*j];
			result.labels.data[result.labels.step[0]*(i+minY) + result.labels.step[1]*(j+minX)] = labels.data[labels.step[0]*i + labels.step[1]*j];
		}
	}	

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
