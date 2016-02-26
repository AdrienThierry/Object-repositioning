#include <stdio.h>
#include <cmath>
#include "GMM.hpp"

struct GMM computeGMM(IplImage *imageIpl) {
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

	// Get probs and labels
	for (int i = 0 ; i < image.rows ; i++) {
		std::vector<double> probsRow;
		std::vector<int> labelsRow;
		for (int j = 0 ; j < image.cols ; j++) {
			labelsRow.push_back((int)(labels.data[labels.step[0]*i + labels.step[1]*j]));
			probsRow.push_back((double)(probs.data[probs.step[0]*i + probs.step[1]*j + labelsRow.back()]) / 255.0);
		}
		result.probs.push_back(probsRow);
		result.labels.push_back(labelsRow);
	}

	// Get GMM weights
	for (int i = 0 ; i < NB_GMM_CLUSTERS ; i++) {
		cv::Mat weights = model.get<cv::Mat>("weights");
		result.weights[i] = (double)(weights.data[weights.step[0]*i + weights.step[1]*0] / 255.0);
	}

	// Compute log likelihoods
	for (int i = 0 ; i < image.rows ; i++) {
		std::vector<double> row;
		for (int j = 0 ; j < image.cols ; j++) {
			row.push_back(
				-1.0 * log(result.probs.at(i).at(j))
				-1.0 * log(result.weights[result.labels.at(i).at(j)]));
		}
		result.weightedLL.push_back(row);
	}

	return result;
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
				newProbsRow.push_back(GMM->probs.at(i-minY).at(j-minX));
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

	GMM->probs = newProbs;
	GMM->labels = newLabels;
	GMM->weightedLL = newWeightedLL;
}


void convertGMMLabelsToCV_Mat(IplImage** result, struct GMM *GMM, int rows, int cols) {
	cv::Mat tmpResult = cv::Mat::zeros(rows, cols, CV_8UC3);

	// Create OpenCV matrix
	for (int i = 0 ; i < tmpResult.rows ; i++) {
		for (int j = 0 ; j < tmpResult.cols ; j++) {
			int color = (int)(255.0 / (double)NB_GMM_CLUSTERS * GMM->labels.at(i).at(j));

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

void convertGMMWeightedLLToCV_Mat(IplImage** result, struct GMM *GMM, int rows, int cols) {
	cv::Mat tmpResult = cv::Mat::zeros(rows, cols, CV_8UC3);

	// Find minimum and maximum of weightedLL for normalization
	double minWeightedLL = GMM->weightedLL.at(0).at(0);
	double maxWeightedLL = GMM->weightedLL.at(0).at(0);
	for (unsigned int i = 0 ; i < GMM->weightedLL.size() ; i++) {
		for (unsigned int j = 0 ; j < GMM->weightedLL.at(0).size() ; j++) {
			if (GMM->weightedLL.at(i).at(j) >= maxWeightedLL) {
				maxWeightedLL = GMM->weightedLL.at(i).at(j);
			}

			if (GMM->weightedLL.at(i).at(j) <= minWeightedLL) {
				minWeightedLL = GMM->weightedLL.at(i).at(j);
			}
		}
	}	

	// Create OpenCV matrix
	for (int i = 0 ; i < tmpResult.rows ; i++) {
		for (int j = 0 ; j < tmpResult.cols ; j++) {
			int color = (int)((GMM->weightedLL.at(i).at(j) - minWeightedLL) * 255.0 / maxWeightedLL);

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
