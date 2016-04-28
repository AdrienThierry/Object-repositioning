#include <stdio.h>
#include <cmath>
#include "GMM.hpp"

void *computeGMM(void *args) {
	
	struct GMM_arg_struct *GMM_args = (struct GMM_arg_struct*)args;

	cv::Mat image(GMM_args->imageIpl, true);

	cv::EM model(NB_GMM_CLUSTERS);
	cv::Mat samples;
	cv::Mat logLikelihood;
	cv::Mat labels;
	cv::Mat probs;

	image.reshape(1,image.rows*image.cols).convertTo(samples,CV_64FC1,1.0/255.0);
	model.train(samples, logLikelihood, labels, probs);

	labels = labels.reshape(1, image.rows);

	// Get GMM weights
	for (int i = 0 ; i < NB_GMM_CLUSTERS ; i++) {
		cv::Mat weights = model.get<cv::Mat>("weights");
		GMM_args->result->weights[i] = weights.at<double>(0,i);
	}

	// Get labels
	for (int i = 0 ; i < image.rows ; i++) {
		std::vector<int> labelsRow;
		for (int j = 0 ; j < image.cols ; j++) {
			labelsRow.push_back((int)(labels.data[labels.step[0]*i + labels.step[1]*j]));
		}
		GMM_args->result->labels.push_back(labelsRow);
	}

	// Compute weighted probs
	for (int i = 0 ; i < image.rows ; i++) {
		std::vector<double> row;
		for (int j = 0 ; j < image.cols ; j++) {
			double currentProb = 0.0;

			for (int k = 0 ; k < NB_GMM_CLUSTERS ; k++) {
				currentProb += GMM_args->result->weights[k]*probs.at<double>(i*image.cols+j, k);
			}

			row.push_back(currentProb);
		}
		GMM_args->result->weightedProbs.push_back(row);
	}

	// Compute log likelihoods
	for (int i = 0 ; i < image.rows ; i++) {
		std::vector<double> row;
		for (int j = 0 ; j < image.cols ; j++) {
			float saliency = GMM_args->saliencyLUT->at(i*image.cols+j);

			// 1 - saliency when background (saliencyLUT is computed differently for background and foreground
			row.push_back(-1.0 * log(GMM_args->result->weightedProbs.at(i).at(j) * (double)saliency));
		}
		GMM_args->result->weightedLL.push_back(row);
	}

	sem_post(GMM_args->semaphore);

	return NULL;
}

IplImage preProcessingForegroundGMM(IplImage *image, BoundingBox bb) {
	// Determine min x, max x, min y and max y in bounding box
	computeBBEnds(&bb);
	int minX = bb.ends[0];
	int maxX = bb.ends[1];
	int minY = bb.ends[2];
	int maxY = bb.ends[3];

	cv::Mat imageMat(image, true);
	cv::Mat resultMat = cv::Mat::zeros(maxY-minY+1, maxX-minX+1, CV_8UC3);

	// Create new image
	for (int i = 0 ; i < resultMat.rows ; i++) {
		for (int j = 0 ; j < resultMat.cols ; j++) {
			resultMat.data[resultMat.step[0]*i + resultMat.step[1]*j + 0] = imageMat.data[imageMat.step[0]*(i+minY) + imageMat.step[1]*(j+minX) + 0];
			resultMat.data[resultMat.step[0]*i + resultMat.step[1]*j + 1] = imageMat.data[imageMat.step[0]*(i+minY) + imageMat.step[1]*(j+minX) + 1];
			resultMat.data[resultMat.step[0]*i + resultMat.step[1]*j + 2] = imageMat.data[imageMat.step[0]*(i+minY) + imageMat.step[1]*(j+minX) + 2];
		}
	}

	IplImage result = resultMat;

	return result;
}

void postProcessingForegroundGMM(struct GMM *GMM, BoundingBox bb, int rows, int cols) {
	// Determine min x, max x, min y and max y in bounding box
	computeBBEnds(&bb);
	int minX = bb.ends[0];
	//int maxX = bb.ends[1];
	int minY = bb.ends[2];
	//int maxY = bb.ends[3];

	std::vector<std::vector<double> > newProbs;
	std::vector<std::vector<int> > newLabels;
	std::vector<std::vector<double> > newWeightedLL;

	for (int i = 0 ; i < rows ; i++) {
		std::vector<double> newProbsRow;
		std::vector<int> newLabelsRow;
		std::vector<double> newWeightedLLRow;
		for (int j = 0 ; j < cols ; j++) {
			Point currentPixel;
			currentPixel.x = j;
			currentPixel.y = i;
			if (isInsideBB(currentPixel, bb)) {
				newProbsRow.push_back(GMM->weightedProbs.at(i-minY).at(j-minX));
				newLabelsRow.push_back(GMM->labels.at(i-minY).at(j-minX));
				newWeightedLLRow.push_back(GMM->weightedLL.at(i-minY).at(j-minX));
			}
			else {
				newProbsRow.push_back(0.0);
				newLabelsRow.push_back(0);
				newWeightedLLRow.push_back(0.0);
			}
		}
		newProbs.push_back(newProbsRow);
		newLabels.push_back(newLabelsRow);
		newWeightedLL.push_back(newWeightedLLRow);
	}

	GMM->weightedProbs = newProbs;
	GMM->labels = newLabels;
	GMM->weightedLL = newWeightedLL;
}

void convertGMMWeightedProbsToCV_Mat(IplImage** result, struct GMM *GMM, int rows, int cols) {
	cv::Mat tmpResult = cv::Mat::zeros(rows, cols, CV_8UC3);

	// Create OpenCV matrix
	for (int i = 0 ; i < tmpResult.rows ; i++) {
		for (int j = 0 ; j < tmpResult.cols ; j++) {
			int color = (int)(255.0 * GMM->weightedProbs.at(i).at(j));

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
