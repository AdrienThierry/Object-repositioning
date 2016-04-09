#include "Depth.hpp"

std::vector<std::vector<float> > computeDepthMap(struct PolygonalChain *groundLine, SDL_Rect* foregroundPosition, IplImage *foreground, int rows, int cols) {
	
	std::vector<std::vector<float> > result;

	cv::Mat foregroundMat(foreground);

	int yGround = groundLine->y;
	int yBottom = foregroundPosition->y + foregroundPosition->h; // Bottom of object

	for (int i = 0 ; i < rows ; i++) {
		std::vector<float> row;

		for (int j = 0 ; j < cols ; j++) {
			// If pixel is foreground
			if (foregroundMat.data[foregroundMat.step[0]*i + foregroundMat.step[1]*j + 3] != 0) {
				row.push_back(((float)yBottom - (float)yGround)/(float)yGround);
			}

			// Else
			else {
				row.push_back(0.0);
			}
		}

		result.push_back(row);
	}	

	return result;
}

void convertDepthToCV_Mat(IplImage **result, std::vector<std::vector<float> > *depthMap) {
	int rows = depthMap->size();
	int cols = depthMap->at(0).size();

	cv::Mat tmpResult = cv::Mat::zeros(rows, cols, CV_8UC3);

	// Create OpenCV matrix
	for (int i = 0 ; i < rows ; i++) {

		for (int j = 0 ; j < cols ; j++) {

			int color = (int) (255.0 * depthMap->at(i).at(j));

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

void computeFinalImage(std::vector<std::vector<float> > *backgroundDepthMap, std::vector<std::vector<float> > *foregroundDepthMap, IplImage **result, IplImage *foreground, IplImage *background, int rows, int cols) {

	cv::Mat tmpResult = cv::Mat::zeros(rows, cols, CV_8UC4);
	cv::Mat foregroundMat(foreground);
	cv::Mat backgroundMat(background);

	// Create OpenCV matrix
	for (int i = 0 ; i < rows ; i++) {

		for (int j = 0 ; j < cols ; j++) {

			if (foregroundDepthMap->at(i).at(j) > backgroundDepthMap->at(i).at(j)) { // Foreground

				tmpResult.data[tmpResult.step[0]*i + tmpResult.step[1]*j + 0] = foregroundMat.data[foregroundMat.step[0]*i + foregroundMat.step[1]*j + 0];
				tmpResult.data[tmpResult.step[0]*i + tmpResult.step[1]*j + 1] = foregroundMat.data[foregroundMat.step[0]*i + foregroundMat.step[1]*j + 1];
				tmpResult.data[tmpResult.step[0]*i + tmpResult.step[1]*j + 2] = foregroundMat.data[foregroundMat.step[0]*i + foregroundMat.step[1]*j + 2];
				tmpResult.data[tmpResult.step[0]*i + tmpResult.step[1]*j + 3] = foregroundMat.data[foregroundMat.step[0]*i + foregroundMat.step[1]*j + 3];

			}

			else { // Background
				tmpResult.data[tmpResult.step[0]*i + tmpResult.step[1]*j + 0] = backgroundMat.data[backgroundMat.step[0]*i + backgroundMat.step[1]*j + 0];
				tmpResult.data[tmpResult.step[0]*i + tmpResult.step[1]*j + 1] = backgroundMat.data[backgroundMat.step[0]*i + backgroundMat.step[1]*j + 1];
				tmpResult.data[tmpResult.step[0]*i + tmpResult.step[1]*j + 2] = backgroundMat.data[backgroundMat.step[0]*i + backgroundMat.step[1]*j + 2];
				tmpResult.data[tmpResult.step[0]*i + tmpResult.step[1]*j + 3] = 255;

			}
		}
	}

	cvReleaseImage(result);
	*result = cvCreateImage(cvSize(cols,rows),8,4);
	IplImage ipltemp = tmpResult;
	cvCopy(&ipltemp,*result);
}
