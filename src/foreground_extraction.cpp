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

void assignSmoothnessTerm(GraphType *g, IplImage *image, int rows, int cols) {
	cv::Mat imageMat(image, true);
	
	// Left-right weights
	for (int i = 0 ; i < rows ; i++) {
		for (int j = 0 ; j < cols-1 ; j++) {
			
			float weight = 0.0;
			struct Color color1, color2;
			struct ColorLab color1Lab, color2Lab;

			color1.r = imageMat.data[imageMat.step[0]*i + imageMat.step[1]*j + 2];
			color1.g = imageMat.data[imageMat.step[0]*i + imageMat.step[1]*j + 1];
			color1.b = imageMat.data[imageMat.step[0]*i + imageMat.step[1]*j + 0];
			
			color2.r = imageMat.data[imageMat.step[0]*i + imageMat.step[1]*(j+1) + 2];
			color2.g = imageMat.data[imageMat.step[0]*i + imageMat.step[1]*(j+1) + 1];
			color2.b = imageMat.data[imageMat.step[0]*i + imageMat.step[1]*(j+1) + 0];

			color1Lab = convertRGB2Lab(color1);
			color2Lab = convertRGB2Lab(color2);

			float dist = distanceLab(color1Lab, color2Lab);

			weight = LAMBDA * exp(-1.0 * dist * dist / (2.0 * (float)SIGMA * (float)SIGMA));

			g -> add_edge( i*cols+j, i*cols+(j+1), /* capacities */ weight , weight );
		}
	}

	// Top-bottom weights
	for (int i = 0 ; i < rows-1 ; i++) {
		for (int j = 0 ; j < cols ; j++) {
			
			float weight = 0.0;
			struct Color color1, color2;
			struct ColorLab color1Lab, color2Lab;

			color1.r = imageMat.data[imageMat.step[0]*i + imageMat.step[1]*j + 2];
			color1.g = imageMat.data[imageMat.step[0]*i + imageMat.step[1]*j + 1];
			color1.b = imageMat.data[imageMat.step[0]*i + imageMat.step[1]*j + 0];
			
			color2.r = imageMat.data[imageMat.step[0]*(i+1) + imageMat.step[1]*j + 2];
			color2.g = imageMat.data[imageMat.step[0]*(i+1) + imageMat.step[1]*j + 1];
			color2.b = imageMat.data[imageMat.step[0]*(i+1) + imageMat.step[1]*j + 0];

			color1Lab = convertRGB2Lab(color1);
			color2Lab = convertRGB2Lab(color2);

			float dist = distanceLab(color1Lab, color2Lab);

			weight = LAMBDA * exp(-1.0 * dist * dist / (2.0 * (float)SIGMA * (float)SIGMA));

			g -> add_edge( i*cols+j, (i+1)*cols+j, /* capacities */ weight , weight );
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
	assignSmoothnessTerm(g, input, rows, cols);

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
