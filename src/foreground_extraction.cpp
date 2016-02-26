#include <opencv2/core/core.hpp>

#include "foreground_extraction.hpp"

GraphType* createGraph(int rows, int cols) {
	GraphType *g = new GraphType(/*estimated # of nodes*/ rows*cols, /*estimated # of edges*/ rows*cols);

	for (int i = 0 ; i < rows*cols ; i++) {
		g -> add_node();
	}

	return g; 
}

void assignDataTerm(GraphType *g, struct GMM* GMMForeground, struct GMM* GMMBackground, struct BoundingBox bb, int rows, int cols) {
	for (int i = 0 ; i < rows ; i++) {
		for (int j = 0 ; j < cols ; j++) {
			struct Point currentPixel;
			currentPixel.x = j;
			currentPixel.y = i;

			double foregroundWeight, backgroundWeight;

			// Possibly foreground
			if (isInsideBB(currentPixel, bb)) {
				// It's normal that background and foreground are inverted (see https://www.youtube.com/watch?v=lYQQ88nzxAM)
				foregroundWeight = GMMBackground->weightedLL.at(i).at(j);
				backgroundWeight = GMMForeground->weightedLL.at(i).at(j);
			}
			// Outside bounding box => positively background
			else {
				foregroundWeight = 0.0;
				backgroundWeight = std::numeric_limits<double>::max();
			}
			g -> add_tweights( i*cols+j,   /* capacities */  foregroundWeight, backgroundWeight );
		}
	}
}

void extractForeground(IplImage *input, IplImage **result, struct GMM* GMMForeground, struct GMM* GMMBackground, struct BoundingBox bb, int rows, int cols) {
	cv::Mat tmpResult = cv::Mat::zeros(rows, cols, CV_8UC3);
	cv::Mat inputMat(input, true);

	// Create graph
	GraphType *g = createGraph(rows, cols);

	// Assign weights
	assignDataTerm(g, GMMForeground, GMMBackground, bb, rows, cols);

	// Perform graph cut
	g -> maxflow();

	// Create resulting image
	for (int i = 0 ; i < rows ; i++) {
		for (int j = 0 ; j < cols ; j++) {
			if (g->what_segment(i*cols+j) == GraphType::SOURCE) { // Foreground
				tmpResult.data[tmpResult.step[0]*i + tmpResult.step[1]*j + 0] = inputMat.data[inputMat.step[0]*i + inputMat.step[1]*j + 0];
				tmpResult.data[tmpResult.step[0]*i + tmpResult.step[1]*j + 1] = inputMat.data[inputMat.step[0]*i + inputMat.step[1]*j + 1];
				tmpResult.data[tmpResult.step[0]*i + tmpResult.step[1]*j + 2] = inputMat.data[inputMat.step[0]*i + inputMat.step[1]*j + 2];
			}

			// If background, pixel is black => Do nothing
		}
	}

	cvReleaseImage(result);
	*result = cvCreateImage(cvSize(cols,rows),8,3);
	IplImage ipltemp = tmpResult;
	cvCopy(&ipltemp,*result);

	freeGraph(g);
}

void freeGraph(GraphType *g) {
	delete g;
}
