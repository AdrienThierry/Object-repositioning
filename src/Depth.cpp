#include "Depth.hpp"

std::vector<std::vector<float> > computeDepthMap(struct PolygonalChain *groundLine, SDL_Rect *foregroundPosition, struct Foreground *foreground, struct BoundingBox *initialBB, int rows, int cols) {
	
	std::vector<std::vector<float> > result;

	int yGround = groundLine->y;
	int yBottom = foregroundPosition->y + foregroundPosition->h; // Bottom of object

	int initBBMinX = initialBB->ends[0];
	int initBBMinY = initialBB->ends[2];

	int BBMinX = foreground->bb.ends[0];
	int BBMinY = foreground->bb.ends[2];	

	int offsetX, offsetY;

	if (BBMinX <= initBBMinX)
		offsetX = initBBMinX - BBMinX;
	else
		offsetX = BBMinX - initBBMinX;

	if (BBMinY <= initBBMinY)
		offsetY = initBBMinY - BBMinY;
	else
		offsetY = BBMinY - initBBMinY;

	printf("%d %d\n", offsetX, offsetY);

	for (int i = 0 ; i < rows ; i++) {
		std::vector<float> row;

		for (int j = 0 ; j < cols ; j++) {
			// If pixel is foreground
			if (i+offsetY < rows && j+offsetX < cols) {

				if (foreground->mask.at(i+offsetY).at(j+offsetX) == true) {
					row.push_back(((float)yBottom - (float)yGround)/(float)yGround);
				}
		
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
