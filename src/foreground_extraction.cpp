#include <opencv2/core/core.hpp>

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

cv::Mat* computeProbsForeground(cv::Mat *image, struct BoundingBox bb) {
	cv::EM model(NB_GMM);
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

	cv::Mat foreground(maxY-minY+1, maxX-minX+1, CV_8UC3);
	for (int i = 0 ; i < foreground.rows ; i++) {
		for (int j = 0 ; j < foreground.cols ; j++) {
			foreground.data[foreground.step[0]*i + foreground.step[1]*j + 0] = image->data[image->step[0]*(i+minY) + image->step[1]*(j+minX) + 0];
			foreground.data[foreground.step[0]*i + foreground.step[1]*j + 1] = image->data[image->step[0]*(i+minY) + image->step[1]*(j+minX) + 1];
			foreground.data[foreground.step[0]*i + foreground.step[1]*j + 2] = image->data[image->step[0]*(i+minY) + image->step[1]*(j+minX) + 2];
		}
	}

	foreground.reshape(1,foreground.rows*foreground.cols).convertTo(samples,CV_64FC1,1.0/255.0);
	model.train(samples, logLikelihood, labels, probs);

	probs = probs.reshape(NB_GMM, foreground.rows);
	labels = labels.reshape(1, foreground.rows);

	// Only return probabilities of foreground (first GMM cluster)
	cv::Mat *result = new cv::Mat(image->rows, image->cols, CV_64FC1);
	*result = cv::Mat::zeros(image->rows, image->cols, CV_64FC1);

	for (int i = 0 ; i < foreground.rows ; i++) {
		for (int j = 0 ; j < foreground.cols ; j++) {
			if (labels.data[labels.step[0]*i + labels.step[1]*j] == 1) {
				result->data[result->step[0]*(i+minY) + result->step[1]*(j+minX)] = probs.data[probs.step[0]*i + probs.step[1]*j + 1];
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
